/*
 * OzaBLAS Scheme I wrapper for rocHPL
 * This file must be compiled with hipcc to access ozablas
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <dlfcn.h>

#ifdef HPL_USE_OZAKI
#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"

extern rocblas_handle handle;
static rocblas_handle& get_global_rocblas_handle() {
    return handle;
}

using rocblas_dgemm_t = rocblas_status (*)(rocblas_handle, rocblas_operation, rocblas_operation,
                                            int, int, int, const double*,
                                            const double*, int, const double*, int,
                                            const double*, double*, int);

static rocblas_dgemm_t get_real_rocblas_dgemm() {
    static rocblas_dgemm_t real_fn = nullptr;
    if (!real_fn) {
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
    rocblas_operation rbOpA = (transa == HIPBLAS_OP_N) ? rocblas_operation_none :
                              (transa == HIPBLAS_OP_T) ? rocblas_operation_transpose :
                              rocblas_operation_conjugate_transpose;
    rocblas_operation rbOpB = (transb == HIPBLAS_OP_N) ? rocblas_operation_none :
                              (transb == HIPBLAS_OP_T) ? rocblas_operation_transpose :
                              rocblas_operation_conjugate_transpose;
    
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

struct OzakiContext {
    std::shared_ptr<ozablas::HipExecutor> executor;
    std::unordered_map<size_t, std::unique_ptr<ozablas::WorkspaceScheme1>> workspaces;
    double* temp_buffer = nullptr;
    size_t temp_buffer_size = 0;
    std::mutex mtx;
    
    OzakiContext() {
        int device;
        hipGetDevice(&device);
        executor = std::make_shared<ozablas::HipExecutor>(device);
    }
    
    ~OzakiContext() {
        if (temp_buffer) {
            hipFree(temp_buffer);
        }
    }
    
    ozablas::WorkspaceScheme1* get_workspace(size_t M, size_t N, size_t K, size_t slices) {
        size_t key = (M << 40) | (N << 20) | K;
        
        std::lock_guard<std::mutex> lock(mtx);
        auto it = workspaces.find(key);
        if (it != workspaces.end() && it->second->get_slices() == slices) {
            return it->second.get();
        }
        
        workspaces[key] = std::make_unique<ozablas::WorkspaceScheme1>(executor, M, N, K, slices);
        return workspaces[key].get();
    }
    
    double* get_temp_buffer(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (temp_buffer_size < size) {
            if (temp_buffer) {
                hipFree(temp_buffer);
            }
            hipError_t err = hipMalloc(&temp_buffer, size);
            if (err != hipSuccess) {
                temp_buffer = nullptr;
                temp_buffer_size = 0;
                return nullptr;
            }
            temp_buffer_size = size;
        }
        return temp_buffer;
    }
};

std::unordered_map<int, std::unique_ptr<OzakiContext>>& get_contexts() {
    static std::unordered_map<int, std::unique_ptr<OzakiContext>> contexts;
    return contexts;
}

std::mutex& get_contexts_mutex() {
    static std::mutex mtx;
    return mtx;
}

OzakiContext* get_context() {
    int device;
    hipGetDevice(&device);
    
    std::lock_guard<std::mutex> lock(get_contexts_mutex());
    auto& contexts = get_contexts();
    if (contexts.find(device) == contexts.end()) {
        contexts[device] = std::make_unique<OzakiContext>();
    }
    return contexts[device].get();
}

int get_num_slices() {
    static int num_slices = -1;
    if (num_slices < 0) {
        const char* env = std::getenv("OZAKI_SLICES");
        num_slices = env ? std::atoi(env) : 0;
    }
    return num_slices;
}

__global__ void scale_and_add_kernel(double* C, const double* temp, double alpha, double beta, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = beta * C[idx] + alpha * temp[idx];
    }
}

__global__ void scale_kernel(double* C, double beta, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = beta * C[idx];
    }
}

__global__ void add_scaled_kernel(double* C, const double* temp, double alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] += alpha * temp[idx];
    }
}

}  // anonymous namespace

extern "C" hipblasStatus_t hpl_ozaki_dgemm(
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
    int num_slices = get_num_slices();
    
    if (num_slices < 2 || m <= 0 || n <= 0 || k <= 0) {
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    double h_alpha, h_beta;
    hipMemcpy(&h_alpha, alpha, sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&h_beta, beta, sizeof(double), hipMemcpyDeviceToHost);
    
    hipStream_t stream;
    hipblasGetStream(hipHandle, &stream);
    
    bool transA = (transa != HIPBLAS_OP_N);
    bool transB = (transb != HIPBLAS_OP_N);
    
    size_t M = static_cast<size_t>(m);
    size_t N = static_cast<size_t>(n);
    size_t K = static_cast<size_t>(k);
    
    OzakiContext* ctx = get_context();
    
    size_t buffer_size = M * N * sizeof(double);
    double* temp = ctx->get_temp_buffer(buffer_size);
    if (!temp) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[OZAKI] Temp buffer allocation failed, falling back to native DGEMM\n");
            logged = true;
        }
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    static int ozaki_call_count = 0;
    ozaki_call_count++;
    if (ozaki_call_count <= 3) {
        fprintf(stderr, "[OZAKI ACTIVE] Call #%d: m=%d, n=%d, k=%d, slices=%d, transA=%d, transB=%d, alpha=%.2f, beta=%.2f\n", 
                ozaki_call_count, m, n, k, num_slices, transA, transB, h_alpha, h_beta);
    }
    
    try {
        ozablas::WorkspaceScheme1* ws = ctx->get_workspace(M, N, K, num_slices);
        
        hipMemset(temp, 0, buffer_size);
        
        ozablas::ozaki_scheme1_gemm(*ws, A, B, temp);
        
        ctx->executor->synchronize();
        
        size_t total_elements = M * N;
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        
        if (h_beta == 0.0) {
            hipLaunchKernelGGL(add_scaled_kernel, dim3(num_blocks), dim3(block_size), 0, stream,
                               C, temp, h_alpha, total_elements);
        } else {
            hipLaunchKernelGGL(scale_and_add_kernel, dim3(num_blocks), dim3(block_size), 0, stream,
                               C, temp, h_alpha, h_beta, total_elements);
        }
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[OZAKI] Exception: %s, falling back to native DGEMM\n", e.what());
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    return HIPBLAS_STATUS_SUCCESS;
}

#endif  // HPL_USE_OZAKI
