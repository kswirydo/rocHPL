#!/bin/bash
# Maximum N Weak Scaling Benchmark Script for rocHPL
# Uses largest possible N values (~216 GB per GPU)
# Compares Native DGEMM vs GEMMul8 (Ozaki-2 INT8 emulation)
#
# Usage: ./run_weak_scaling_maxN.sh [--native-only | --gemmul8-only]

set -e

# Configuration
ROCHPL_DIR="/home/kswirydo/rocHPL"
BUILD_DIR="${ROCHPL_DIR}/build/rocHPL_gemmul8"
GEMMUL8_DIR="/home/kswirydo/GEMMul8/GEMMul8"

# Environment setup
export PATH="${ROCHPL_DIR}/tpl/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="${ROCHPL_DIR}/tpl/openmpi/lib:${ROCHPL_DIR}/tpl/ucx/lib:${GEMMUL8_DIR}/lib:$LD_LIBRARY_PATH"

# Maximum N values (~216 GB per GPU, ~75% of 288GB MI355X memory)
# N scales with sqrt(num_gpus) to keep per-GPU work constant
NB=384

# Maximum N values for native DGEMM
N1_NATIVE=98304    # 1 GPU: ~216 GB
N2_NATIVE=138624   # 2 GPU: ~215 GB per GPU  
N4_NATIVE=196608   # 4 GPU: ~216 GB per GPU
N8_NATIVE=277632   # 8 GPU: ~215 GB per GPU

# For GEMMul8, use smaller N due to workspace requirements
# GEMMul8 workspace ≈ 2*m*n*num_moduli bytes for intermediate results
# Use ~60% of native N to leave room for workspace
N1_GEMMUL8=64896   # 1 GPU: ~94 GB + workspace
N2_GEMMUL8=91392   # 2 GPU: ~93 GB per GPU + workspace
N4_GEMMUL8=129792  # 4 GPU: ~94 GB per GPU + workspace
N8_GEMMUL8=183168  # 8 GPU: ~94 GB per GPU + workspace

# GEMMul8 moduli count
GEMMUL8_MODULI=12

# Results file
RESULTS_FILE="weak_scaling_maxN_$(date +%Y%m%d_%H%M%S).txt"

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

print_header "Maximum N Weak Scaling Benchmark"
echo ""
echo "Native DGEMM N values: 1GPU=$N1_NATIVE, 2GPU=$N2_NATIVE, 4GPU=$N4_NATIVE, 8GPU=$N8_NATIVE"
echo "GEMMul8 N values:      1GPU=$N1_GEMMUL8, 2GPU=$N2_GEMMUL8, 4GPU=$N4_GEMMUL8, 8GPU=$N8_GEMMUL8"
echo "Block size: $NB"
echo ""
echo "Results file: $RESULTS_FILE"
echo ""

echo "mode,gpus,P,Q,N,NB,time_sec,gflops,status" > "$RESULTS_FILE"

if [ "$RUN_NATIVE" == "true" ]; then
    print_header "Native DGEMM (Maximum N, ~216 GB per GPU)"
    echo ""
    run_test 1 1 1 $N1_NATIVE "native"
    run_test 2 1 2 $N2_NATIVE "native"
    run_test 4 2 2 $N4_NATIVE "native"
    run_test 8 2 4 $N8_NATIVE "native"
    echo ""
fi

if [ "$RUN_GEMMUL8" == "true" ]; then
    print_header "GEMMul8 (${GEMMUL8_MODULI} moduli, ~94 GB per GPU + workspace)"
    echo ""
    run_test 1 1 1 $N1_GEMMUL8 "gemmul8"
    run_test 2 1 2 $N2_GEMMUL8 "gemmul8"
    run_test 4 2 2 $N4_GEMMUL8 "gemmul8"
    run_test 8 2 4 $N8_GEMMUL8 "gemmul8"
    echo ""
fi

print_header "Results Summary"
echo ""
cat "$RESULTS_FILE" | column -t -s','
echo ""

if [ "$RUN_NATIVE" == "true" ] && [ "$RUN_GEMMUL8" == "true" ]; then
    print_header "Performance Comparison (Note: Different N values)"
    echo ""
    echo "GPUs | Native (N)      | TFlops | GEMMul8 (N)     | TFlops | Ratio"
    echo "-----|-----------------|--------|-----------------|--------|-------"
    
    for gpus in 1 2 4 8; do
        native_line=$(grep "^native,${gpus}," "$RESULTS_FILE")
        gemmul8_line=$(grep "^gemmul8,${gpus}," "$RESULTS_FILE")
        
        if [ -n "$native_line" ] && [ -n "$gemmul8_line" ]; then
            native_n=$(echo "$native_line" | cut -d',' -f5)
            native_gflops=$(echo "$native_line" | cut -d',' -f8)
            gemmul8_n=$(echo "$gemmul8_line" | cut -d',' -f5)
            gemmul8_gflops=$(echo "$gemmul8_line" | cut -d',' -f8)
            
            native_tflops=$(echo "$native_gflops" | awk '{printf "%.1f", $1/1000}')
            gemmul8_tflops=$(echo "$gemmul8_gflops" | awk '{printf "%.1f", $1/1000}')
            ratio=$(echo "$gemmul8_gflops $native_gflops" | awk '{printf "%.1f%%", ($1/$2)*100}')
            
            printf "%4d | N=%-13s | %6s | N=%-13s | %6s | %s\n" "$gpus" "$native_n" "$native_tflops" "$gemmul8_n" "$gemmul8_tflops" "$ratio"
        fi
    done
    echo ""
fi

echo "Benchmark complete!"
