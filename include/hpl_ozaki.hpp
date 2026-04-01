/*
 * OzaBLAS Scheme I integration for rocHPL
 * Provides FP64 DGEMM emulation using INT8 tensor cores via Ozaki Scheme I
 */

#ifndef HPL_OZAKI_HPP
#define HPL_OZAKI_HPP

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdlib>
#include <cstdio>

#ifdef HPL_USE_OZAKI
extern "C" hipblasStatus_t hpl_ozaki_dgemm(
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

namespace hpl_ozaki {

inline int get_num_slices() {
    static int num_slices = -1;
    if (num_slices < 0) {
        const char* env = std::getenv("OZAKI_SLICES");
        num_slices = env ? std::atoi(env) : 0;
        if (num_slices > 0) {
            fprintf(stderr, "[rocHPL+OzaBLAS] Enabled Scheme I with %d slices\n", num_slices);
        }
    }
    return num_slices;
}

inline hipblasStatus_t dgemm_ozaki(
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
#ifdef HPL_USE_OZAKI
    int num_slices = get_num_slices();
    if (num_slices >= 2 && m > 0 && n > 0 && k > 0) {
        return hpl_ozaki_dgemm(handle, transa, transb, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif
    return hipblasDgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace hpl_ozaki

#define HPL_DGEMM_OZAKI(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    hpl_ozaki::dgemm_ozaki(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

#ifdef HPL_USE_OZAKI
#ifdef HPL_DGEMM
#undef HPL_DGEMM
#endif
#define HPL_DGEMM(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    hpl_ozaki::dgemm_ozaki(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#endif

#endif  // HPL_OZAKI_HPP
