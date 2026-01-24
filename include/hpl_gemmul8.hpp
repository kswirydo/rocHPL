/*
 * GEMMul8 integration for rocHPL
 * Provides FP64 DGEMM emulation using INT8 tensor cores
 */

#ifndef HPL_GEMMUL8_HPP
#define HPL_GEMMUL8_HPP

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdlib>
#include <cstdio>

#ifdef HPL_USE_GEMMUL8
// C-style wrapper function implemented in HPL_gemmul8_wrapper.cpp
// This allows linking with libgemmul8.a without needing hipcc for all files
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
);
#endif

namespace hpl_gemmul8 {

// Configuration from environment
inline int get_num_moduli() {
    static int num_moduli = -1;
    if (num_moduli < 0) {
        const char* env = std::getenv("GEMMUL8_NUM_MOD_D");
        num_moduli = env ? std::atoi(env) : 0;
        if (num_moduli > 0) {
            fprintf(stderr, "[rocHPL+GEMMul8] Enabled with %d moduli\n", num_moduli);
        }
    }
    return num_moduli;
}

// GEMMul8-accelerated DGEMM wrapper
inline hipblasStatus_t dgemm_gemmul8(
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
#ifdef HPL_USE_GEMMUL8
    int num_moduli = get_num_moduli();
    if (num_moduli >= 2 && m > 0 && n > 0 && k > 0) {
        return hpl_gemmul8_dgemm(handle, transa, transb, m, n, k,
                                 alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif
    // Fall back to native DGEMM
    return hipblasDgemm(handle, transa, transb, m, n, k,
                       alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace hpl_gemmul8

// Macro for easy replacement in source files
#define HPL_DGEMM(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    hpl_gemmul8::dgemm_gemmul8(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

#endif  // HPL_GEMMUL8_HPP
