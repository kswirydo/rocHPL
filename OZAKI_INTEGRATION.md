# rocHPL + OzaBLAS Integration

This document describes the integration of OzaBLAS (Ozaki Scheme I) with rocHPL for potential FP64 GEMM acceleration using INT8 Tensor Cores.

## Repository Branches

| Repository | Branch | Description |
|------------|--------|-------------|
| `/home/kswirydo/rocHPL` | `ozaki_one_tests` | rocHPL with OzaBLAS wrapper integration |
| `/home/kswirydo/ozablas` | `column-wise` | OzaBLAS converted to column-major storage |

## Overview of Changes

### OzaBLAS Changes (`column-wise` branch)

The original OzaBLAS library used **row-major** matrix storage, but rocHPL (like most BLAS libraries) uses **column-major** storage. The `column-wise` branch converts all OzaBLAS kernels to column-major.

#### 1. Kernel Indexing (all pipeline files)

**Before (row-major):**
```cpp
size_t idx = row * cols + col;
double v = A[row * cols + col];
```

**After (column-major):**
```cpp
size_t idx = row + col * rows;
double v = A[row + col * rows];
```

Files modified:
- `src/pipeline/step1_statistics.hpp` - shift/scale factor computation
- `src/pipeline/step2_slicing.hpp` - INT8 slice extraction  
- `src/pipeline/step4_reconstruction.hpp` - FP64 result reconstruction

#### 2. rocBLAS GEMM Parameters (`src/hip/pipeline.hip`)

**Before (row-major trick using C=B*A):**
```cpp
rocblas_gemm_ex(handle, op_none, op_none,
    N, M, K, &alpha,
    B_t, i8, N,    // B as "first" matrix
    A_t, i8, K,    // A as "second" matrix
    ...);
```

**After (column-major C=A*B):**
```cpp
rocblas_gemm_ex(handle, op_none, op_none,
    M, N, K, &alpha,
    A_t, i8, M,    // A(M×K), leading dim = M
    B_t, i8, K,    // B(K×N), leading dim = K
    ...);
```

### rocHPL Changes (`ozaki_one_tests` branch)

#### 1. OzaBLAS Wrapper (`src/HPL_ozaki_wrapper_device.cpp`)

A wrapper that intercepts `hipblasDgemm` calls and dispatches to OzaBLAS when appropriate:

- Handles transpose operations (HPL often uses `transB=T`)
- Transposes B matrix using `rocblas_dgeam` when needed
- Copies non-contiguous matrices to contiguous buffers
- Applies alpha/beta scaling after OzaBLAS computation
- Falls back to native rocBLAS DGEMM for unsupported cases

#### 2. Memory Management

The wrapper includes careful memory management to prevent OOM:

- **No workspace caching**: Creates fresh workspace per GEMM call and frees immediately
- **Smart temp buffer caching**: Buffers >1GB are not cached; allocated fresh and freed after use
- **Accurate memory estimation**: Accounts for all allocations before proceeding
- **Configurable limits**: `OZAKI_MAX_WORKSPACE_GB` environment variable (default: 16GB)

#### 3. Build System (`install.sh`)

Added `--with-ozaki` flag to enable OzaBLAS integration:
```bash
./install.sh --with-ozaki=/path/to/ozablas
```

## Building

### 1. Build OzaBLAS (column-major branch)

```bash
cd /home/kswirydo/ozablas
git checkout column-wise
make release-hip
```

### 2. Build rocHPL with OzaBLAS

```bash
cd /home/kswirydo/rocHPL
git checkout ozaki_one_tests
./install.sh --with-ozaki=/home/kswirydo/ozablas
```

Or rebuild in the build directory:
```bash
cd /home/kswirydo/rocHPL/build
make -j16
```

## Running

### With OzaBLAS (Ozaki Scheme I)

```bash
# Enable with 3 slices (recommended starting point)
OZAKI_SLICES=3 ./mpirun_rochpl -P 1 -Q 1 -N 64128 --NB 512

# With custom memory limit
OZAKI_SLICES=3 OZAKI_MAX_WORKSPACE_GB=32 ./mpirun_rochpl -P 1 -Q 1 -N 64128 --NB 512
```

### Without OzaBLAS (native rocBLAS DGEMM)

```bash
# Don't set OZAKI_SLICES, or set to 0 or 1
./mpirun_rochpl -P 1 -Q 1 -N 64128 --NB 512

# Or explicitly disable
OZAKI_SLICES=0 ./mpirun_rochpl -P 1 -Q 1 -N 64128 --NB 512
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OZAKI_SLICES` | 0 (disabled) | Number of INT8 slices. Set ≥2 to enable Ozaki. More slices = better accuracy, worse performance |
| `OZAKI_MAX_WORKSPACE_GB` | 16 | Maximum workspace memory in GB. GEMMs exceeding this fall back to native DGEMM |

## Performance Considerations

### Slice Count Tradeoff

- **Fewer slices (2-3)**: Faster but less accurate
- **More slices (4-8)**: More accurate but slower (Scheme I scales as O(S²))

### When Ozaki Falls Back to Native DGEMM

1. `OZAKI_SLICES < 2`
2. `transA != OP_N` (transposed A not yet supported)
3. Workspace memory estimate exceeds `OZAKI_MAX_WORKSPACE_GB`
4. Any allocation failure

### Memory Usage

For a GEMM with dimensions M×N×K and S slices, approximate memory:
- `d_A_slices`: S × M × K bytes
- `d_B_slices`: S × K × N bytes  
- `d_C_tc`: M × N × 4 bytes
- `temp_C`: M × N × 8 bytes
- `temp_A/B`: up to M×K×8 + K×N×8 bytes (if copies needed)

## Diagnostics

The wrapper prints diagnostic messages to stderr:

```
[rocHPL+OzaBLAS] Enabled Scheme I with 3 slices
[OZAKI #1] m=64096 n=32 k=32 transB=1 | Running totals: OP_T=1, OP_N=0
[OZAKI] GEMM too large (m=90000, n=90000, k=896, workspace=98.5 GB, limit=16 GB), using native DGEMM
```

At the end of the run:
```
[OZAKI STATS] Ozaki GEMMs: 1234, dgeam (transpose B): 1200, fallback dgemm: 56, copy A: 0, copy B: 34
```

## Known Limitations

1. **transA not supported**: GEMMs with transposed A fall back to native DGEMM
2. **Large GEMMs**: Very large trailing updates may exceed memory limits and fall back
3. **Accuracy**: With few slices, accuracy may be lower than native FP64 DGEMM
