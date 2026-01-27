#!/bin/bash
# Strong Scaling Benchmark Script for rocHPL
# Compares Native DGEMM vs GEMMul8 (Ozaki-2 INT8 emulation)
# Fixed problem size, varying number of GPUs
#
# Usage: ./run_strong_scaling.sh [--native-only | --gemmul8-only] [N]

set -e

# Configuration
ROCHPL_DIR="/home/kswirydo/rocHPL"
BUILD_DIR="${ROCHPL_DIR}/build/rocHPL_gemmul8"
GEMMUL8_DIR="/home/kswirydo/GEMMul8/GEMMul8"

# Environment setup
export PATH="${ROCHPL_DIR}/tpl/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="${ROCHPL_DIR}/tpl/openmpi/lib:${ROCHPL_DIR}/tpl/ucx/lib:${GEMMUL8_DIR}/lib:$LD_LIBRARY_PATH"

# Default problem size (can be overridden via command line)
NB=384
N=${2:-45312}  # Default N, or take from second argument

# GEMMul8 moduli count
GEMMUL8_MODULI=12

# Results file
RESULTS_FILE="strong_scaling_N${N}_$(date +%Y%m%d_%H%M%S).txt"

cd "$BUILD_DIR"

# Ensure symlink exists
mkdir -p rocHPL/bin 2>/dev/null || true
ln -sf "$(pwd)/bin/rochpl" rocHPL/bin/rochpl 2>/dev/null || true

print_header() {
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

run_test() {
    local gpus=$1
    local p=$2
    local q=$3
    local n=$4
    local mode=$5
    
    if [ "$mode" == "gemmul8" ]; then
        export GEMMUL8_NUM_MOD_D=$GEMMUL8_MODULI
    else
        unset GEMMUL8_NUM_MOD_D
    fi
    
    echo "Running: ${gpus} GPU(s), P=${p}, Q=${q}, N=${n}, NB=${NB}, Mode=${mode}"
    
    ./mpirun_rochpl -P $p -Q $q -N $n --NB $NB 2>&1 >/dev/null
    
    result=$(grep "WC10" HPL.out | tail -1)
    status=$(grep -E "PASSED|FAILED" HPL.out | tail -1)
    
    time=$(echo "$result" | awk '{print $6}')
    gflops=$(echo "$result" | awk '{print $7}')
    pass_fail=$(echo "$status" | grep -o "PASSED\|FAILED")
    
    printf "  -> Time: %8.2f s, Performance: %12s GFlops, Status: %s\n" "$time" "$gflops" "$pass_fail"
    
    echo "${mode},${gpus},${p},${q},${n},${time},${gflops},${pass_fail}" >> "$RESULTS_FILE"
}

# Parse arguments
RUN_NATIVE=true
RUN_GEMMUL8=true

if [ "$1" == "--native-only" ]; then
    RUN_GEMMUL8=false
elif [ "$1" == "--gemmul8-only" ]; then
    RUN_NATIVE=false
fi

print_header "Strong Scaling Benchmark (Fixed N=$N)"
echo ""
echo "Problem size N: $N"
echo "Block size NB:  $NB"
echo "Results file:   $RESULTS_FILE"
echo ""

echo "mode,gpus,P,Q,N,time_sec,gflops,status" > "$RESULTS_FILE"

if [ "$RUN_NATIVE" == "true" ]; then
    print_header "Native DGEMM"
    echo ""
    run_test 1 1 1 $N "native"
    run_test 2 1 2 $N "native"
    run_test 4 2 2 $N "native"
    run_test 8 2 4 $N "native"
    echo ""
fi

if [ "$RUN_GEMMUL8" == "true" ]; then
    print_header "GEMMul8 (${GEMMUL8_MODULI} moduli)"
    echo ""
    run_test 1 1 1 $N "gemmul8"
    run_test 2 1 2 $N "gemmul8"
    run_test 4 2 2 $N "gemmul8"
    run_test 8 2 4 $N "gemmul8"
    echo ""
fi

print_header "Results"
echo ""
cat "$RESULTS_FILE" | column -t -s','
echo ""
echo "Benchmark complete!"
