/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <algorithm>

rocblas_handle  handle;
hipblasHandle_t hipblasHandle;

hipStream_t computeStream, dataStream;

hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];
hipEvent_t pfactStart, pfactStop;

static char host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void HPL_InitGPU(const HPL_T_grid* GRID) {
  char host_name[MPI_MAX_PROCESSOR_NAME];

  int i, n, namelen, rank, nprocs;
  int dev;

  int nprow, npcol, myrow, mycol;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(host_name, &namelen);

  int localSize = GRID->local_npcol * GRID->local_nprow;
  int localRank = rank % localSize;

  /* Find out how many GPUs are in the system and their device number */
  int deviceCount;
  CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

  if(deviceCount < 1) {
    if(localRank == 0)
      HPL_pwarn(stderr,
                __LINE__,
                "HPL_InitGPU",
                "Node %s found no GPUs. Is the ROCm kernel module loaded?",
                host_name);
    MPI_Finalize();
    exit(1);
  }

  dev = localRank % deviceCount;

  /* Assign device to MPI process, initialize BLAS and probe device properties
   */
  CHECK_HIP_ERROR(hipSetDevice(dev));

  CHECK_HIP_ERROR(hipStreamCreate(&computeStream));
  CHECK_HIP_ERROR(hipStreamCreate(&dataStream));

  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(update + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(update + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(update + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(&pfactStart));
  CHECK_HIP_ERROR(hipEventCreate(&pfactStop));

  /* Create a rocBLAS handle */
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
  CHECK_ROCBLAS_ERROR(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
  CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, computeStream));

  rocblas_initialize();

  /* Create a hipBLAS handle (for DGEMM interception by GEMMul8) */
  CHECK_HIPBLAS_ERROR(hipblasCreate(&hipblasHandle));
  CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(hipblasHandle, HIPBLAS_POINTER_MODE_HOST));
  CHECK_HIPBLAS_ERROR(hipblasSetStream(hipblasHandle, computeStream));

  /* Warmup hipblasGemmEx with INT8 to pre-initialize hipblaslt */
  {
    int8_t *warmupA = nullptr, *warmupB = nullptr;
    int32_t *warmupC = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&warmupA, 16 * 16 * sizeof(int8_t)));
    CHECK_HIP_ERROR(hipMalloc(&warmupB, 16 * 16 * sizeof(int8_t)));
    CHECK_HIP_ERROR(hipMalloc(&warmupC, 16 * 16 * sizeof(int32_t)));
    CHECK_HIP_ERROR(hipMemset(warmupA, 0, 16 * 16 * sizeof(int8_t)));
    CHECK_HIP_ERROR(hipMemset(warmupB, 0, 16 * 16 * sizeof(int8_t)));
    CHECK_HIP_ERROR(hipMemset(warmupC, 0, 16 * 16 * sizeof(int32_t)));
    int32_t alpha = 1, beta = 0;
    hipblasStatus_t gemmStatus = hipblasGemmEx(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                  16, 16, 16, &alpha,
                  warmupA, HIP_R_8I, 16,
                  warmupB, HIP_R_8I, 16,
                  &beta,
                  warmupC, HIP_R_32I, 16,
                  HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
    if (gemmStatus != HIPBLAS_STATUS_SUCCESS) {
      std::cerr << "hipblasGemmEx INT8 warmup failed with status " << gemmStatus << std::endl;
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipFree(warmupA));
    CHECK_HIP_ERROR(hipFree(warmupB));
    CHECK_HIP_ERROR(hipFree(warmupC));
  }

#ifdef HPL_ROCBLAS_ALLOW_ATOMICS
  CHECK_ROCBLAS_ERROR(
      rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));
#else
  CHECK_ROCBLAS_ERROR(
      rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));
#endif
}

void HPL_FreeGPU() {
  CHECK_HIPBLAS_ERROR(hipblasDestroy(hipblasHandle));
  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(pfactStart));
  CHECK_HIP_ERROR(hipEventDestroy(pfactStop));

  CHECK_HIP_ERROR(hipStreamDestroy(computeStream));
  CHECK_HIP_ERROR(hipStreamDestroy(dataStream));
}
