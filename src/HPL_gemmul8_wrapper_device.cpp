/*
 * GEMMul8 wrapper for rocHPL
 * This file must be compiled with hipcc to access gemmul8 templates
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>

#ifdef HPL_USE_GEMMUL8
#include "gemmul8.hpp"
#include "self_hipify.hpp"  // For gemmul8_rocblas::set_external_handle

// rocHPL's global rocblas handle (declared in HPL_InitGPU.cpp as 'handle')
extern rocblas_handle handle;
static rocblas_handle& get_global_rocblas_handle() {
    return handle;
}

// Get the real rocblas_dgemm bypassing any hooks
using rocblas_dgemm_t = rocblas_status (*)(rocblas_handle, rocblas_operation, rocblas_operation,
                                            int, int, int, const double*,
                                            const double*, int, const double*, int,
                                            const double*, double*, int);
static rocblas_dgemm_t get_real_rocblas_dgemm() {
    static rocblas_dgemm_t real_fn = nullptr;
    if (!real_fn) {
        // Try to find librocblas.so already loaded
        void* lib = dlopen("librocblas.so", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) {
            lib = dlopen("/opt/rocm/lib/librocblas.so", RTLD_NOW);
        }
        if (!lib) {
            lib = dlopen("librocblas.so", RTLD_NOW);
        }
        if (lib) {
            real_fn = (rocblas_dgemm_t)dlsym(lib, "rocblas_dgemm");
        }
    }
    return real_fn;
}

// Helper to call real rocblas_dgemm with hipblas handle
static hipblasStatus_t call_real_dgemm(
    hipblasHandle_t hipHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    // Use the global rocblas handle instead of converting hipblas handle
    rocblas_operation rbOpA = (transa == HIPBLAS_OP_N) ? rocblas_operation_none :
                              (transa == HIPBLAS_OP_T) ? rocblas_operation_transpose :
                              rocblas_operation_conjugate_transpose;
    rocblas_operation rbOpB = (transb == HIPBLAS_OP_N) ? rocblas_operation_none :
                              (transb == HIPBLAS_OP_T) ? rocblas_operation_transpose :
                              rocblas_operation_conjugate_transpose;
    
    // Sync stream
    hipStream_t stream;
    hipblasGetStream(hipHandle, &stream);
    rocblas_set_stream(get_global_rocblas_handle(), stream);
    
    auto real_dgemm = get_real_rocblas_dgemm();
    if (!real_dgemm) {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    
    rocblas_status st = real_dgemm(get_global_rocblas_handle(), rbOpA, rbOpB, 
                                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return (st == rocblas_status_success) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_EXECUTION_FAILED;
}

namespace {

// Workspace management - single global workspace per device
struct WorkspaceManager {
    std::unordered_map<int, std::pair<void*, size_t>> device_workspaces;
    std::mutex mtx;
    
    void* get(size_t required_size) {
        int device;
        hipGetDevice(&device);
        
        std::lock_guard<std::mutex> lock(mtx);
        auto& ws = device_workspaces[device];
        if (ws.second < required_size) {
            if (ws.first) {
                hipFree(ws.first);
                ws.first = nullptr;
            }
            hipError_t err = hipMalloc(&ws.first, required_size);
            if (err != hipSuccess) {
                // Clear the error state so it doesn't affect later operations
                hipGetLastError();  // Consume the error
                ws.first = nullptr;
                ws.second = 0;
                return nullptr;
            }
            ws.second = required_size;
        }
        return ws.first;
    }
    
    ~WorkspaceManager() {
        for (auto& kv : device_workspaces) {
            if (kv.second.first) hipFree(kv.second.first);
        }
    }
};

WorkspaceManager& get_workspace_manager() {
    static WorkspaceManager mgr;
    return mgr;
}

bool is_fastmode() {
    static int fastmode = -1;
    if (fastmode < 0) {
        const char* env = std::getenv("GEMMUL8_FASTMODE");
        fastmode = env ? std::atoi(env) : 1;  // Default to fast mode
    }
    return fastmode != 0;
}

}  // anonymous namespace

extern "C" hipblasStatus_t hpl_gemmul8_dgemm(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
) {
    // Get num_moduli from environment
    static int num_moduli = -1;
    if (num_moduli < 0) {
        const char* env = std::getenv("GEMMUL8_NUM_MOD_D");
        num_moduli = env ? std::atoi(env) : 0;
    }
    
    if (num_moduli < 2 || m <= 0 || n <= 0 || k <= 0) {
        // Call the real rocblas_dgemm bypassing any hooks
        return call_real_dgemm(handle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // Calculate workspace size - always use UseExtraWorkspace=true
    size_t worksize = gemmul8::workSize<false, true>(
        static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(k), 
        static_cast<unsigned>(num_moduli), false, false, nullptr, nullptr);
    
    
    void* workspace = get_workspace_manager().get(worksize);
    if (workspace == nullptr) {
        // Fallback to native DGEMM if workspace allocation fails
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[GEMMUL8] Workspace too large (%.2f GB), using native DGEMM for large GEMMs\n",
                    worksize / 1e9);
            logged = true;
        }
        return call_real_dgemm(handle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // Initialize gemmul8 to use rocHPL's rocblas handle (only once)
    static bool handle_set = false;
    if (!handle_set) {
        gemmul8_rocblas::set_external_handle(get_global_rocblas_handle());
        handle_set = true;
    }
    
    // Debug: confirm GEMMul8 is being called
    static int gemmul8_call_count = 0;
    gemmul8_call_count++;
    if (gemmul8_call_count <= 3) {
        fprintf(stderr, "[GEMMUL8 ACTIVE] Call #%d: m=%d, n=%d, k=%d, moduli=%d\n", 
                gemmul8_call_count, m, n, k, num_moduli);
    }
    
    // Call gemmul8
    gemmul8::gemm<double, true>(
        handle,
        transa, transb,
        static_cast<size_t>(m), 
        static_cast<size_t>(n), 
        static_cast<size_t>(k),
        alpha, A, static_cast<size_t>(lda),
        B, static_cast<size_t>(ldb),
        beta, C, static_cast<size_t>(ldc),
        static_cast<unsigned>(num_moduli),
        is_fastmode(),
        workspace,
        nullptr, nullptr,
        false, false,
        false, false
    );
    
    // Note: No stream sync here - rocHPL manages synchronization
    return HIPBLAS_STATUS_SUCCESS;
}

#endif  // HPL_USE_GEMMUL8
