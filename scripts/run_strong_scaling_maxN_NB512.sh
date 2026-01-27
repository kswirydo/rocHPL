#!/bin/bash
# Maximum N Strong Scaling with NB=512
# Fixed large N, varying number of GPUs, with NB=512
# Compares Native DGEMM vs GEMMul8 (Ozaki-2 INT8 emulation)
#
# Usage: ./run_strong_scaling_maxN_NB512.sh [--native-only | --gemmul8-only]

set -e

# Configuration
ROCHPL_DIR="/home/kswirydo/rocHPL"
BUILD_DIR="${ROCHPL_DIR}/build/rocHPL_gemmul8"
GEMMUL8_DIR="/home/kswirydo/GEMMul8/GEMMul8"

# Environment setup
export PATH="${ROCHPL_DIR}/tpl/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="${ROCHPL_DIR}/tpl/openmpi/lib:${ROCHPL_DIR}/tpl/ucx/lib:${GEMMUL8_DIR}/lib:$LD_LIBRARY_PATH"

# Block size
NB=512

# Fixed N values (must be multiples of NB=512)
N_NATIVE=98304   # 192*512, ~216 GB for 1 GPU
N_GEMMUL8=64512  # 126*512, ~93 GB for 1 GPU + workspace

# GEMMul8 moduli count
GEMMUL8_MODULI=12

# Results file
RESULTS_FILE="strong_scaling_maxN_NB${NB}_$(date +%Y%m%d_%H%M%S).txt"

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
    
    echo "${mode},${gpus},${p},${q},${n},${NB},${time},${gflops},${pass_fail}" >> "$RESULTS_FILE"
}

# Parse arguments
RUN_NATIVE=true
RUN_GEMMUL8=true

if [ "$1" == "--native-only" ]; then
    RUN_GEMMUL8=false
elif [ "$1" == "--gemmul8-only" ]; then
    RUN_NATIVE=false
fi

print_header "Maximum N Strong Scaling with NB=${NB}"
echo ""
echo "Native DGEMM: N=$N_NATIVE (fixed)"
echo "GEMMul8:      N=$N_GEMMUL8 (fixed)"
echo "Block size:   NB=$NB"
echo ""
echo "Results file: $RESULTS_FILE"
echo ""

echo "mode,gpus,P,Q,N,NB,time_sec,gflops,status" > "$RESULTS_FILE"

if [ "$RUN_NATIVE" == "true" ]; then
    print_header "Native DGEMM (N=$N_NATIVE, NB=$NB)"
    echo ""
    run_test 1 1 1 $N_NATIVE "native"
    run_test 2 1 2 $N_NATIVE "native"
    run_test 4 2 2 $N_NATIVE "native"
    run_test 8 2 4 $N_NATIVE "native"
    echo ""
fi

if [ "$RUN_GEMMUL8" == "true" ]; then
    print_header "GEMMul8 (N=$N_GEMMUL8, NB=$NB, ${GEMMUL8_MODULI} moduli)"
    echo ""
    run_test 1 1 1 $N_GEMMUL8 "gemmul8"
    run_test 2 1 2 $N_GEMMUL8 "gemmul8"
    run_test 4 2 2 $N_GEMMUL8 "gemmul8"
    run_test 8 2 4 $N_GEMMUL8 "gemmul8"
    echo ""
fi

print_header "Results Summary"
echo ""
cat "$RESULTS_FILE" | column -t -s','
echo ""

# Calculate speedup for strong scaling
if [ "$RUN_NATIVE" == "true" ]; then
    echo ""
    print_header "Native DGEMM Strong Scaling Efficiency"
    echo ""
    base_time=$(grep "^native,1," "$RESULTS_FILE" | cut -d',' -f7)
    echo "GPUs | Time (s) | Speedup | Efficiency"
    echo "-----|----------|---------|----------"
    for gpus in 1 2 4 8; do
        time=$(grep "^native,${gpus}," "$RESULTS_FILE" | cut -d',' -f7)
        if [ -n "$time" ] && [ -n "$base_time" ]; then
            speedup=$(echo "$base_time $time" | awk '{printf "%.2f", $1/$2}')
            efficiency=$(echo "$speedup $gpus" | awk '{printf "%.1f%%", ($1/$2)*100}')
            printf "%4d | %8s | %7s | %s\n" "$gpus" "$time" "${speedup}x" "$efficiency"
        fi
    done
    echo ""
fi

if [ "$RUN_GEMMUL8" == "true" ]; then
    echo ""
    print_header "GEMMul8 Strong Scaling Efficiency"
    echo ""
    base_time=$(grep "^gemmul8,1," "$RESULTS_FILE" | cut -d',' -f7)
    echo "GPUs | Time (s) | Speedup | Efficiency"
    echo "-----|----------|---------|----------"
    for gpus in 1 2 4 8; do
        time=$(grep "^gemmul8,${gpus}," "$RESULTS_FILE" | cut -d',' -f7)
        if [ -n "$time" ] && [ -n "$base_time" ]; then
            speedup=$(echo "$base_time $time" | awk '{printf "%.2f", $1/$2}')
            efficiency=$(echo "$speedup $gpus" | awk '{printf "%.1f%%", ($1/$2)*100}')
            printf "%4d | %8s | %7s | %s\n" "$gpus" "$time" "${speedup}x" "$efficiency"
        fi
    done
    echo ""
fi

if [ "$RUN_NATIVE" == "true" ] && [ "$RUN_GEMMUL8" == "true" ]; then
    echo ""
    print_header "Performance Comparison: Native vs GEMMul8"
    echo ""
    echo "Note: Different N values (Native=$N_NATIVE, GEMMul8=$N_GEMMUL8)"
    echo ""
    echo "GPUs | Native TFlops | GEMMul8 TFlops | Ratio (GEMMul8/Native)"
    echo "-----|---------------|----------------|------------------------"
    
    for gpus in 1 2 4 8; do
        native_gflops=$(grep "^native,${gpus}," "$RESULTS_FILE" | cut -d',' -f8)
        gemmul8_gflops=$(grep "^gemmul8,${gpus}," "$RESULTS_FILE" | cut -d',' -f8)
        
        if [ -n "$native_gflops" ] && [ -n "$gemmul8_gflops" ]; then
            native_tflops=$(echo "$native_gflops" | awk '{printf "%.1f", $1/1000}')
            gemmul8_tflops=$(echo "$gemmul8_gflops" | awk '{printf "%.1f", $1/1000}')
            ratio=$(echo "$gemmul8_gflops $native_gflops" | awk '{printf "%.1f%%", ($1/$2)*100}')
            
            printf "%4d | %13s | %14s | %s\n" "$gpus" "$native_tflops" "$gemmul8_tflops" "$ratio"
        fi
    done
    echo ""
fi

echo "Benchmark complete!"
