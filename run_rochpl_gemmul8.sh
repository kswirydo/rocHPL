#!/bin/bash
# rocHPL wrapper script with GEMMul8 INT8 emulated DGEMM
# This script intercepts hipBLAS DGEMM calls and replaces them with
# GEMMul8's INT8 tensor core emulation

#===========================================
# GEMMul8 Configuration
#===========================================
GEMMUL8_LIB="/home/kswirydo/GEMMul8/GEMMul8/lib/libgemmul8.so"

# Number of moduli for DGEMM (2-20, higher = more accurate but slower)
# Recommended: 15-18 for double precision accuracy
export GEMMUL8_NUM_MOD_D=${GEMMUL8_NUM_MOD_D:-15}

# Fast mode (1) or accurate mode (0)
# Fast mode is faster but slightly less accurate
export GEMMUL8_FASTMODE_D=${GEMMUL8_FASTMODE_D:-1}

# Preallocate workspace for large matrices (avoids reallocation during run)
# Set these based on your HPL problem size
export GEMMUL8_MAX_M=${GEMMUL8_MAX_M:-32768}
export GEMMUL8_MAX_N=${GEMMUL8_MAX_N:-32768}
export GEMMUL8_MAX_K=${GEMMUL8_MAX_K:-32768}
export GEMMUL8_MAX_NUM_MOD=${GEMMUL8_MAX_NUM_MOD:-20}

# Enable scaling skip for repeated matrices (performance optimization)
export GEMMUL8_SKIP_SCALE_A=${GEMMUL8_SKIP_SCALE_A:-0}
export GEMMUL8_SKIP_SCALE_B=${GEMMUL8_SKIP_SCALE_B:-0}

# Use extra workspace for better performance
export GEMMUL8_USE_EXTRA_WORKSPACE=${GEMMUL8_USE_EXTRA_WORKSPACE:-1}

#===========================================
# Verify GEMMul8 library exists
#===========================================
if [ ! -f "$GEMMUL8_LIB" ]; then
    echo "ERROR: GEMMul8 library not found at $GEMMUL8_LIB"
    echo "Please build GEMMul8 first:"
    echo "  cd /home/kswirydo/GEMMul8/GEMMul8 && make -j8"
    exit 1
fi

#===========================================
# Print configuration
#===========================================
echo "=========================================="
echo "rocHPL with GEMMul8 INT8 Emulation"
echo "=========================================="
echo "GEMMul8 Library: $GEMMUL8_LIB"
echo "DGEMM Configuration:"
echo "  - Num Moduli: $GEMMUL8_NUM_MOD_D"
echo "  - Fast Mode: $GEMMUL8_FASTMODE_D"
echo "  - Max M/N/K: $GEMMUL8_MAX_M / $GEMMUL8_MAX_N / $GEMMUL8_MAX_K"
echo "  - Max Num Moduli: $GEMMUL8_MAX_NUM_MOD"
echo "  - Skip Scale A/B: $GEMMUL8_SKIP_SCALE_A / $GEMMUL8_SKIP_SCALE_B"
echo "  - Extra Workspace: $GEMMUL8_USE_EXTRA_WORKSPACE"
echo "=========================================="
echo ""

#===========================================
# Run rocHPL directly with LD_PRELOAD
#===========================================
# For single GPU runs, we run rochpl directly with LD_PRELOAD
# This avoids issues with LD_PRELOAD affecting MPI setup scripts

ROCHPL_BIN="/home/kswirydo/rocHPL/build/bin/rochpl"
ROCM_DIR="/opt/rocm"
ROCHPL_WORKDIR="/home/kswirydo/rocHPL/build/rocHPL_gemmul8"

# Set library paths
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:${LD_LIBRARY_PATH}"

# Change to working directory with HPL.dat
if [ ! -f "$ROCHPL_WORKDIR/HPL.dat" ]; then
    echo "ERROR: HPL.dat not found at $ROCHPL_WORKDIR/HPL.dat"
    echo "Please create an HPL.dat configuration file in that directory."
    exit 1
fi

cd "$ROCHPL_WORKDIR"
echo "Working directory: $(pwd)"
echo ""

# Run with LD_PRELOAD for GEMMul8
exec env LD_PRELOAD="$GEMMUL8_LIB" "$ROCHPL_BIN" "$@"
