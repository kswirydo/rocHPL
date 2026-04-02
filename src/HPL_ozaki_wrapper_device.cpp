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
    std::unordered_map<size_t, std::unique_ptr<ozablas::WorkspaceScheme1>> workspaces;
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
    
    double* get_temp_C(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (temp_C_size < size) {
            if (temp_C) hipFree(temp_C);
            if (hipMalloc(&temp_C, size) != hipSuccess) {
                temp_C = nullptr;
                temp_C_size = 0;
                return nullptr;
            }
            temp_C_size = size;
        }
        return temp_C;
    }
    
    double* get_temp_B(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (temp_B_size < size) {
            if (temp_B) hipFree(temp_B);
            if (hipMalloc(&temp_B, size) != hipSuccess) {
                temp_B = nullptr;
                temp_B_size = 0;
                return nullptr;
            }
            temp_B_size = size;
        }
        return temp_B;
    }
    
    double* get_temp_A(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (temp_A_size < size) {
            if (temp_A) hipFree(temp_A);
            if (hipMalloc(&temp_A, size) != hipSuccess) {
                temp_A = nullptr;
                temp_A_size = 0;
                return nullptr;
            }
            temp_A_size = size;
        }
        return temp_A;
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
    // Scheme I workspace scales as O(M*N*slices) which can be huge
    size_t workspace_estimate = (size_t)m * n * num_slices * sizeof(int32_t);
    const size_t MAX_WORKSPACE = 8ULL * 1024 * 1024 * 1024;  // 8 GB limit
    if (workspace_estimate > MAX_WORKSPACE) {
        static bool logged = false;
        if (!logged) {
            fprintf(stderr, "[OZAKI] GEMM too large (m=%d, n=%d, estimated workspace=%.1f GB), using native DGEMM\n",
                    m, n, workspace_estimate / 1e9);
            logged = true;
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
    
    
    try {
        hipGetLastError();
        
        // Allocate temp buffer for ozablas output (contiguous M x N)
        size_t temp_C_bytes = M * N * sizeof(double);
        double* temp_C = ctx->get_temp_C(temp_C_bytes);
        if (!temp_C) {
            return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        
        const double* A_ozaki = A;
        const double* B_ozaki = B;
        
        // Handle A: if lda != M, need to copy to contiguous buffer
        double* temp_A = nullptr;
        if (lda != (int)M) {
            size_t temp_A_bytes = M * K * sizeof(double);
            temp_A = ctx->get_temp_A(temp_A_bytes);
            if (!temp_A) {
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            
            dim3 block(16, 16);
            dim3 grid((M + 15) / 16, (K + 15) / 16);
            hipLaunchKernelGGL(copy_matrix_kernel, grid, block, 0, stream,
                               temp_A, M, A, lda, M, K);
            A_ozaki = temp_A;
            get_stats().copy_A_count++;
        }
        
        // Handle B based on transpose
        double* temp_B = nullptr;
        size_t B_rows_ozaki, B_cols_ozaki;
        
        if (transB) {
            // B is N x K (column-major), we need B^T which is K x N
            // ozablas needs K x N contiguous column-major
            B_rows_ozaki = K;
            B_cols_ozaki = N;
            
            size_t temp_B_bytes = K * N * sizeof(double);
            temp_B = ctx->get_temp_B(temp_B_bytes);
            if (!temp_B) {
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            
            // Transpose B (N x K with ldb) -> temp_B (K x N contiguous)
            // Use rocblas_dgeam for efficient transpose
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
                temp_B, K
            );
            if (st != rocblas_status_success) {
                get_stats().fallback_dgemm_count++;
                return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                      alpha, A, lda, B, ldb, beta, C, ldc);
            }
            B_ozaki = temp_B;
            get_stats().dgeam_count++;
        } else {
            // B is K x N (column-major), use directly if contiguous
            B_rows_ozaki = K;
            B_cols_ozaki = N;
            
            if (ldb != (int)K) {
                size_t temp_B_bytes = K * N * sizeof(double);
                temp_B = ctx->get_temp_B(temp_B_bytes);
                if (!temp_B) {
                    return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
                }
                
                dim3 block(16, 16);
                dim3 grid((K + 15) / 16, (N + 15) / 16);
                hipLaunchKernelGGL(copy_matrix_kernel, grid, block, 0, stream,
                                   temp_B, K, B, ldb, K, N);
                B_ozaki = temp_B;
                get_stats().copy_B_count++;
            }
        }
        
        hipStreamSynchronize(stream);
        
        // Get workspace and compute
        ozablas::WorkspaceScheme1* ws = ctx->get_workspace(M, N, K, num_slices);
        
        hipMemsetAsync(temp_C, 0, temp_C_bytes, stream);
        hipStreamSynchronize(stream);
        
        // ozablas computes: temp_C = A_ozaki * B_ozaki
        // A_ozaki is M x K, B_ozaki is K x N, temp_C is M x N (all contiguous column-major)
        ozablas::ozaki_scheme1_gemm(*ws, A_ozaki, B_ozaki, temp_C);
        get_stats().ozaki_gemm_count++;
        
        ctx->executor->synchronize();
        
        hipError_t ozaki_err = hipGetLastError();
        if (ozaki_err != hipSuccess) {
            fprintf(stderr, "[OZAKI] HIP error after gemm: %s\n", hipGetErrorString(ozaki_err));
            return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        
        // Apply alpha/beta: C = beta * C + alpha * temp_C
        // Handle non-contiguous C (ldc != M)
        dim3 block(16, 16);
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        hipLaunchKernelGGL(fused_scale_add_kernel, grid, block, 0, stream,
                           C, ldc, temp_C, M, h_alpha, h_beta, M, N);
        
        hipError_t kernel_err = hipGetLastError();
        if (kernel_err != hipSuccess) {
            fprintf(stderr, "[OZAKI] HIP error in scale kernel: %s\n", hipGetErrorString(kernel_err));
            return HIPBLAS_STATUS_EXECUTION_FAILED;
        }
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[OZAKI] Exception: %s\n", e.what());
        return call_real_dgemm(hipHandle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    return HIPBLAS_STATUS_SUCCESS;
}

#endif  // HPL_USE_OZAKI
