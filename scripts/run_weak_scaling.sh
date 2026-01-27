#!/bin/bash
# Weak Scaling Benchmark Script for rocHPL
# Compares Native DGEMM vs GEMMul8 (Ozaki-2 INT8 emulation)
#
# Usage: ./run_weak_scaling.sh [--native-only | --gemmul8-only]

set -e

# Configuration
ROCHPL_DIR="/home/kswirydo/rocHPL"
BUILD_DIR="${ROCHPL_DIR}/build/rocHPL_gemmul8"
GEMMUL8_DIR="/home/kswirydo/GEMMul8/GEMMul8"

# Environment setup
export PATH="${ROCHPL_DIR}/tpl/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="${ROCHPL_DIR}/tpl/openmpi/lib:${ROCHPL_DIR}/tpl/ucx/lib:${GEMMUL8_DIR}/lib:$LD_LIBRARY_PATH"

# Weak scaling parameters
# N scales with sqrt(num_gpus) to keep per-GPU work constant
NB=384
BASE_N=64896  # Base problem size for 1 GPU (~94 GB per GPU)

# Calculate N for each GPU count (must be multiple of NB)
N1=$BASE_N
N2=$(( (BASE_N * 1414 / 1000 / NB) * NB ))  # sqrt(2) ≈ 1.414
N4=$(( (BASE_N * 2 / NB) * NB ))             # sqrt(4) = 2
N8=$(( (BASE_N * 2828 / 1000 / NB) * NB ))  # sqrt(8) ≈ 2.828

# GEMMul8 moduli count (12 is sufficient for HPL accuracy requirements)
GEMMUL8_MODULI=12

# Results file
RESULTS_FILE="weak_scaling_results_$(date +%Y%m%d_%H%M%S).txt"

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
    local mode=$5  # "native" or "gemmul8"
    
    if [ "$mode" == "gemmul8" ]; then
        export GEMMUL8_NUM_MOD_D=$GEMMUL8_MODULI
    else
        unset GEMMUL8_NUM_MOD_D
    fi
    
    echo "Running: ${gpus} GPU(s), P=${p}, Q=${q}, N=${n}, NB=${NB}, Mode=${mode}"
    
    ./mpirun_rochpl -P $p -Q $q -N $n --NB $NB 2>&1 >/dev/null
    
    # Extract results
    result=$(grep "WC10" HPL.out | tail -1)
    status=$(grep -E "PASSED|FAILED" HPL.out | tail -1)
    
    # Parse time and gflops
    time=$(echo "$result" | awk '{print $6}')
    gflops=$(echo "$result" | awk '{print $7}')
    pass_fail=$(echo "$status" | grep -o "PASSED\|FAILED")
    
    printf "  -> Time: %8.2f s, Performance: %12s GFlops, Status: %s\n" "$time" "$gflops" "$pass_fail"
    
    # Save to results file
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

# Print configuration
print_header "Weak Scaling Benchmark Configuration"
echo ""
echo "Base N (1 GPU): $N1"
echo "N for 2 GPUs:   $N2"
echo "N for 4 GPUs:   $N4"
echo "N for 8 GPUs:   $N8"
echo "Block size:     $NB"
echo ""
echo "Estimated memory per GPU: ~94 GB"
echo ""
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Initialize results file with header
echo "mode,gpus,P,Q,N,time_sec,gflops,status" > "$RESULTS_FILE"

# Run Native DGEMM tests
if [ "$RUN_NATIVE" == "true" ]; then
    print_header "Native DGEMM (FP64 Tensor Cores)"
    echo ""
    
    run_test 1 1 1 $N1 "native"
    run_test 2 1 2 $N2 "native"
    run_test 4 2 2 $N4 "native"
    run_test 8 2 4 $N8 "native"
    
    echo ""
fi

# Run GEMMul8 tests
if [ "$RUN_GEMMUL8" == "true" ]; then
    print_header "GEMMul8 (Ozaki-2 INT8 Emulation, ${GEMMUL8_MODULI} moduli)"
    echo ""
    
    run_test 1 1 1 $N1 "gemmul8"
    run_test 2 1 2 $N2 "gemmul8"
    run_test 4 2 2 $N4 "gemmul8"
    run_test 8 2 4 $N8 "gemmul8"
    
    echo ""
fi

# Print summary
print_header "Results Summary"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE" | column -t -s','
echo ""

# Calculate and display scaling efficiency if both were run
if [ "$RUN_NATIVE" == "true" ] && [ "$RUN_GEMMUL8" == "true" ]; then
    echo ""
    print_header "Performance Comparison"
    echo ""
    echo "GPUs | Native TFlops | GEMMul8 TFlops | Ratio"
    echo "-----|---------------|----------------|------"
    
    for gpus in 1 2 4 8; do
        native_gflops=$(grep "^native,${gpus}," "$RESULTS_FILE" | cut -d',' -f7)
        gemmul8_gflops=$(grep "^gemmul8,${gpus}," "$RESULTS_FILE" | cut -d',' -f7)
        
        if [ -n "$native_gflops" ] && [ -n "$gemmul8_gflops" ]; then
            # Convert scientific notation and calculate ratio
            native_tflops=$(echo "$native_gflops" | awk '{printf "%.1f", $1/1000}')
            gemmul8_tflops=$(echo "$gemmul8_gflops" | awk '{printf "%.1f", $1/1000}')
            ratio=$(echo "$gemmul8_gflops $native_gflops" | awk '{printf "%.1f%%", ($1/$2)*100}')
            
            printf "%4d | %13s | %14s | %s\n" "$gpus" "$native_tflops" "$gemmul8_tflops" "$ratio"
        fi
    done
    echo ""
fi

echo "Benchmark complete!"
