/*
 * OzaBLAS Scheme I wrapper for rocHPL (column-major version)
 * This file must be compiled with hipcc to access ozablas
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <mutex>
#include <atomic>
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
    // No workspace caching - create and destroy per call to avoid memory accumulation
    double* temp_C = nullptr;
    size_t temp_C_size = 0;
    double* temp_B = nullptr;
    size_t temp_B_size = 0;
    double* temp_A = nullptr;
    size_t temp_A_size = 0;
    std::mutex mtx;
    
    OzakiContext() {
        int device;
        hipGetDevice(&device);
        executor = std::make_shared<ozablas::HipExecutor>(device);
    }
    
    ~OzakiContext() {
        if (temp_C) hipFree(temp_C);
        if (temp_B) hipFree(temp_B);
        if (temp_A) hipFree(temp_A);
    }
    
    // Create workspace on demand - caller is responsible for lifetime
    std::unique_ptr<ozablas::WorkspaceScheme1> create_workspace(size_t M, size_t N, size_t K, size_t slices) {
        return std::make_unique<ozablas::WorkspaceScheme1>(executor, M, N, K, slices);
    }
    
    // Max size for cached temp buffers (1 GB each)
    static constexpr size_t MAX_CACHED_TEMP_SIZE = 1024ULL * 1024 * 1024;
    
    double* get_temp_C(size_t size, bool cache = true) {
        std::lock_guard<std::mutex> lock(mtx);
        
        // If request is too large to cache, always allocate fresh
        if (size > MAX_CACHED_TEMP_SIZE) {
            double* ptr = nullptr;
            if (hipMalloc(&ptr, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            return ptr;
        }
        
        if (temp_C_size < size) {
            if (temp_C) {
                hipFree(temp_C);
                temp_C = nullptr;
                temp_C_size = 0;
            }
            if (hipMalloc(&temp_C, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            temp_C_size = size;
        }
        return temp_C;
    }
    
    void free_temp_C_if_not_cached(double* ptr, size_t size) {
        if (size > MAX_CACHED_TEMP_SIZE && ptr) {
            hipFree(ptr);
        }
    }
    
    double* get_temp_B(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (size > MAX_CACHED_TEMP_SIZE) {
            double* ptr = nullptr;
            if (hipMalloc(&ptr, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            return ptr;
        }
        
        if (temp_B_size < size) {
            if (temp_B) {
                hipFree(temp_B);
                temp_B = nullptr;
                temp_B_size = 0;
            }
            if (hipMalloc(&temp_B, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            temp_B_size = size;
        }
        return temp_B;
    }
    
    void free_temp_B_if_not_cached(double* ptr, size_t size) {
        if (size > MAX_CACHED_TEMP_SIZE && ptr) {
            hipFree(ptr);
        }
    }
    
    double* get_temp_A(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (size > MAX_CACHED_TEMP_SIZE) {
            double* ptr = nullptr;
            if (hipMalloc(&ptr, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            return ptr;
        }
        
        if (temp_A_size < size) {
            if (temp_A) {
                hipFree(temp_A);
                temp_A = nullptr;
                temp_A_size = 0;
            }
            if (hipMalloc(&temp_A, size) != hipSuccess) {
                hipGetLastError();
                return nullptr;
            }
            temp_A_size = size;
        }
        return temp_A;
    }
    
    void free_temp_A_if_not_cached(double* ptr, size_t size) {
        if (size > MAX_CACHED_TEMP_SIZE && ptr) {
            hipFree(ptr);
        }
    }
    
    void free_temp_buffers() {
        std::lock_guard<std::mutex> lock(mtx);
        if (temp_C) { hipFree(temp_C); temp_C = nullptr; temp_C_size = 0; }
        if (temp_B) { hipFree(temp_B); temp_B = nullptr; temp_B_size = 0; }
        if (temp_A) { hipFree(temp_A); temp_A = nullptr; temp_A_size = 0; }
    }
    
    size_t get_cached_memory() const {
        return temp_C_size + temp_B_size + temp_A_size;
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

struct OzakiStats {
    std::atomic<int> ozaki_gemm_count{0};
    std::atomic<int> dgeam_count{0};
    std::atomic<int> fallback_dgemm_count{0};
    std::atomic<int> copy_A_count{0};
    std::atomic<int> copy_B_count{0};
    
    ~OzakiStats() {
        if (ozaki_gemm_count > 0 || fallback_dgemm_count > 0) {
            fprintf(stderr, "\n[OZAKI STATS] Ozaki GEMMs: %d, dgeam (transpose B): %d, fallback dgemm: %d, copy A: %d, copy B: %d\n",
                    ozaki_gemm_count.load(), dgeam_count.load(), fallback_dgemm_count.load(),
                    copy_A_count.load(), copy_B_count.load());
        }
    }
};

OzakiStats& get_stats() {
    static OzakiStats stats;
    return stats;
}

__global__ void fused_scale_add_kernel(
    double* __restrict__ C, int ldc,
    const double* __restrict__ temp, int ld_temp,
    double alpha, double beta,
    int M, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < M && j < N) {
        size_t idx_C = (size_t)j * ldc + i;
        size_t idx_temp = (size_t)j * ld_temp + i;
        C[idx_C] = beta * C[idx_C] + alpha * temp[idx_temp];
    }
}

__global__ void copy_matrix_kernel(
    double* __restrict__ dst, int ld_dst,
    const double* __restrict__ src, int ld_src,
    int rows, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        dst[(size_t)j * ld_dst + i] = src[(size_t)j * ld_src + i];
    }
}

__global__ void transpose_matrix_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src, int ld_src,
    int rows, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        dst[(size_t)i * rows + j] = src[(size_t)j * ld_src + i];
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
    
    double h_alpha = *alpha;
    double h_beta = *beta;
    
    hipStream_t stream;
    hipblasGetStream(hipHandle, &stream);
    
    bool transA = (transa != HIPBLAS_OP_N);
    bool transB = (transb != HIPBLAS_OP_N);
    
    // For now, only handle OP_N for A (most common in HPL)
    if (transA) {
        get_stats().fallback_dgemm_count++;
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // Fall back to native DGEMM for very large GEMMs to avoid workspace OOM
    // Calculate actual workspace memory requirements:
    // - d_shift_A: M * 4 bytes
    // - d_shift_B: N * 4 bytes  
    // - d_A_slices: slices * M * K bytes
    // - d_B_slices: slices * K * N bytes
    // - d_C_tc: M * N * 4 bytes
    // - temp_C: M * N * 8 bytes
    // - temp_A/B: up to M*K*8 + K*N*8 bytes
    static size_t max_workspace = 0;
    if (max_workspace == 0) {
        const char* env = std::getenv("OZAKI_MAX_WORKSPACE_GB");
        size_t limit_gb = env ? std::atoi(env) : 16;  // Default 16 GB (conservative)
        max_workspace = limit_gb * 1024ULL * 1024 * 1024;
    }
    
    size_t workspace_estimate = 
        (size_t)m * 4 +                              // d_shift_A
        (size_t)n * 4 +                              // d_shift_B
        (size_t)num_slices * m * k +                 // d_A_slices
        (size_t)num_slices * k * n +                 // d_B_slices
        (size_t)m * n * 4 +                          // d_C_tc
        (size_t)m * n * 8 +                          // temp_C
        (size_t)m * k * 8 +                          // temp_A (worst case)
        (size_t)k * n * 8;                           // temp_B (worst case)
    
    if (workspace_estimate > max_workspace) {
        static int skip_count = 0;
        skip_count++;
        if (skip_count <= 3 || skip_count % 100 == 0) {
            fprintf(stderr, "[OZAKI] GEMM too large (m=%d, n=%d, k=%d, workspace=%.2f GB, limit=%.0f GB), using native DGEMM (skip #%d)\n",
                    m, n, k, workspace_estimate / 1e9, max_workspace / 1e9, skip_count);
        }
        get_stats().fallback_dgemm_count++;
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    size_t M = static_cast<size_t>(m);
    size_t N = static_cast<size_t>(n);
    size_t K = static_cast<size_t>(k);
    
    OzakiContext* ctx = get_context();
    
    static int ozaki_call_count = 0;
    static int transB_T_count = 0;
    static int transB_N_count = 0;
    ozaki_call_count++;
    if (transB) transB_T_count++; else transB_N_count++;
    
    if (ozaki_call_count <= 5 || ozaki_call_count % 20 == 0) {
        fprintf(stderr, "[OZAKI #%d] m=%d n=%d k=%d transB=%d | Running totals: OP_T=%d, OP_N=%d\n", 
                ozaki_call_count, m, n, k, transB, transB_T_count, transB_N_count);
    }
    
    
    // Track allocated temp buffer sizes for cleanup
    double* local_temp_C = nullptr;
    double* local_temp_A = nullptr;
    double* local_temp_B = nullptr;
    size_t local_temp_C_bytes = 0;
    size_t local_temp_A_bytes = 0;
    size_t local_temp_B_bytes = 0;
    
    auto cleanup_temps = [&]() {
        ctx->free_temp_C_if_not_cached(local_temp_C, local_temp_C_bytes);
        ctx->free_temp_A_if_not_cached(local_temp_A, local_temp_A_bytes);
        ctx->free_temp_B_if_not_cached(local_temp_B, local_temp_B_bytes);
    };
    
    try {
        hipGetLastError();
        
        // Allocate temp buffer for ozablas output (contiguous M x N)
        local_temp_C_bytes = M * N * sizeof(double);
        local_temp_C = ctx->get_temp_C(local_temp_C_bytes);
        if (!local_temp_C) {
            cleanup_temps();
            return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        
        const double* A_ozaki = A;
        const double* B_ozaki = B;
        
        // Handle A: if lda != M, need to copy to contiguous buffer
        if (lda != (int)M) {
            local_temp_A_bytes = M * K * sizeof(double);
            local_temp_A = ctx->get_temp_A(local_temp_A_bytes);
            if (!local_temp_A) {
                cleanup_temps();
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            
            dim3 block(16, 16);
            dim3 grid((M + 15) / 16, (K + 15) / 16);
            hipLaunchKernelGGL(copy_matrix_kernel, grid, block, 0, stream,
                               local_temp_A, M, A, lda, M, K);
            A_ozaki = local_temp_A;
            get_stats().copy_A_count++;
        }
        
        // Handle B based on transpose
        if (transB) {
            // B is N x K (column-major), we need B^T which is K x N
            local_temp_B_bytes = K * N * sizeof(double);
            local_temp_B = ctx->get_temp_B(local_temp_B_bytes);
            if (!local_temp_B) {
                cleanup_temps();
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            
            // Transpose B using rocblas_dgeam
            const double one = 1.0;
            const double zero = 0.0;
            rocblas_set_stream(get_global_rocblas_handle(), stream);
            rocblas_status st = rocblas_dgeam(
                get_global_rocblas_handle(),
                rocblas_operation_transpose,
                rocblas_operation_none,
                K, N,
                &one, B, ldb,
                &zero, B, ldb,
                local_temp_B, K
            );
            if (st != rocblas_status_success) {
                cleanup_temps();
                get_stats().fallback_dgemm_count++;
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            B_ozaki = local_temp_B;
            get_stats().dgeam_count++;
        } else {
            // B is K x N (column-major), use directly if contiguous
            if (ldb != (int)K) {
                local_temp_B_bytes = K * N * sizeof(double);
                local_temp_B = ctx->get_temp_B(local_temp_B_bytes);
                if (!local_temp_B) {
                    cleanup_temps();
                    return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
                }
                
                dim3 block(16, 16);
                dim3 grid((K + 15) / 16, (N + 15) / 16);
                hipLaunchKernelGGL(copy_matrix_kernel, grid, block, 0, stream,
                                   local_temp_B, K, B, ldb, K, N);
                B_ozaki = local_temp_B;
                get_stats().copy_B_count++;
            }
        }
        
        hipStreamSynchronize(stream);
        
        // Create workspace for this GEMM - freed when scope exits
        auto ws = ctx->create_workspace(M, N, K, num_slices);
        if (!ws) {
            fprintf(stderr, "[OZAKI] Failed to create workspace (M=%zu, N=%zu, K=%zu)\n", M, N, K);
            cleanup_temps();
            return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        
        hipMemsetAsync(local_temp_C, 0, local_temp_C_bytes, stream);
        hipStreamSynchronize(stream);
        
        // ozablas computes: temp_C = A_ozaki * B_ozaki
        ozablas::ozaki_scheme1_gemm(*ws, A_ozaki, B_ozaki, local_temp_C);
        get_stats().ozaki_gemm_count++;
        
        ctx->executor->synchronize();
        // ws goes out of scope here and frees workspace memory
        
        hipError_t ozaki_err = hipGetLastError();
        if (ozaki_err != hipSuccess) {
            fprintf(stderr, "[OZAKI] HIP error after gemm: %s\n", hipGetErrorString(ozaki_err));
            cleanup_temps();
            return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        
        // Apply alpha/beta: C = beta * C + alpha * temp_C
        dim3 block(16, 16);
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        hipLaunchKernelGGL(fused_scale_add_kernel, grid, block, 0, stream,
                           C, ldc, local_temp_C, M, h_alpha, h_beta, M, N);
        
        hipError_t kernel_err = hipGetLastError();
        if (kernel_err != hipSuccess) {
            fprintf(stderr, "[OZAKI] HIP error in scale kernel: %s\n", hipGetErrorString(kernel_err));
            cleanup_temps();
            return HIPBLAS_STATUS_EXECUTION_FAILED;
        }
        
        // Cleanup non-cached temp buffers
        cleanup_temps();
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[OZAKI] Exception: %s\n", e.what());
        cleanup_temps();
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    return HIPBLAS_STATUS_SUCCESS;
}

#endif  // HPL_USE_OZAKI
