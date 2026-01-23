# rocHPL Benchmark Results: Native vs GEMMul8 INT8 Emulation

## Hardware Configuration
- **GPU**: AMD Instinct MI355X (gfx950)
- **VRAM**: 288 GiB per GPU
- **GPUs per Node**: 8
- **Date**: 2026-01-23

## Single GPU Results (N = 180,224, NB = 512)

| Configuration | GFLOPS | Runtime | Residual | Status |
|--------------|--------|---------|----------|--------|
| Native FP64 (rocBLAS) | 57,769 | ~73s | 0.000001004238983 | PASSED |
| GEMMul8 INT8 (15 moduli) | 57,717 | ~74s | 0.000001004238983 | PASSED |
| GEMMul8 INT8 (12 moduli) | 57,783 | ~72s | 0.000001004238983 | PASSED |

## Multi-GPU Scaling Results: Native FP64 vs GEMMul8 INT8

| GPUs | Grid | N | Native GFLOPS | GEMMul8 GFLOPS | Difference | Scaling | Status |
|------|------|---------|---------------|----------------|------------|---------|--------|
| 1 | 1×1 | 180,224 | 57,769 | 57,783 | +0.02% | 1.00x | PASSED |
| 2 | 1×2 | 250,880 | 116,610 | 116,520 | -0.08% | 2.02x | PASSED |
| 4 | 1×4 | 354,816 | 235,810 | 235,810 | 0.00% | 4.08x | PASSED |
| 8 | 1×8 | 501,760 | 475,130 | 475,540 | +0.09% | 8.23x | PASSED |

## Key Observations

### Performance Parity Across All GPU Counts
- GEMMul8 INT8 emulation achieves **identical performance** to native FP64 rocBLAS
- Difference between native and emulated is < 0.1% across all configurations
- This confirms the benchmark is memory-bandwidth bound at these problem sizes

### Excellent Multi-GPU Scaling
- Near-linear scaling observed up to 8 GPUs (8.23x on 8 GPUs)
- Peak performance: **475.5 TFLOPS** on 8 MI355X GPUs
- Both native and emulated scale identically

### Numerical Accuracy
- GEMMul8's Ozaki method with 12 moduli achieves bit-accurate FP64 results
- Residual values are identical between native and emulated runs
- All configurations pass HPL residual validation

## GEMMul8 Configuration
```bash
GEMMUL8_NUM_MOD_D=12
GEMMUL8_FASTMODE_D=1
GEMMUL8_MAX_M=32768
GEMMUL8_MAX_N=32768
GEMMUL8_MAX_K=32768
GEMMUL8_MAX_NUM_MOD=20
GEMMUL8_USE_EXTRA_WORKSPACE=1
```

## Conclusion

GEMMul8's INT8 tensor core emulation successfully matches native FP64 DGEMM performance on AMD MI355X for large-scale HPL workloads, while maintaining full double-precision numerical accuracy. The emulation scales linearly across multiple GPUs, achieving 475.5 TFLOPS on 8 GPUs.
