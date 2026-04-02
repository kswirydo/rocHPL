#!/bin/bash
# Benchmark script comparing rocHPL with native DGEMM vs Ozaki/GEMMul8
# Runs on 1, 2, 4, and 8 GPUs

# Don't exit on errors - some runs may fail and we want to continue
set +e

#===========================================
# Configuration
#===========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/rocHPL_gemmul8"
MPI_BIN="${SCRIPT_DIR}/tpl/openmpi/bin/mpiexec"
ROCHPL_BIN="${BUILD_DIR}/rocHPL/bin/rochpl"
RUN_SCRIPT="${BUILD_DIR}/run_rochpl"

# Problem size configuration
# N_1GPU: Base matrix size for 1 GPU (will scale with sqrt(num_gpus) for more GPUs)
# For Ozaki/GEMMul8, N must keep local GEMM dimensions <= GEMMUL8_MAX (65536)
# Max N per GPU ≈ 65536, so N_1GPU=65280 is safe (multiple of NB=640)
# This scales to: 2 GPU→92160, 4 GPU→130560, 8 GPU→184640
N_1GPU=${N_1GPU:-65280}
NB=${NB:-640}

# GEMMul8 settings
export GEMMUL8_NUM_MOD_D=${GEMMUL8_NUM_MOD_D:-15}
export GEMMUL8_FASTMODE_D=${GEMMUL8_FASTMODE_D:-1}
export GEMMUL8_MAX_M=${GEMMUL8_MAX_M:-65536}
export GEMMUL8_MAX_N=${GEMMUL8_MAX_N:-65536}
export GEMMUL8_MAX_K=${GEMMUL8_MAX_K:-65536}
export GEMMUL8_MAX_NUM_MOD=${GEMMUL8_MAX_NUM_MOD:-20}
export GEMMUL8_USE_EXTRA_WORKSPACE=${GEMMUL8_USE_EXTRA_WORKSPACE:-1}

# Results file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${BUILD_DIR}/ozaki_comparison_${TIMESTAMP}.txt"

# Library paths
export LD_LIBRARY_PATH="${SCRIPT_DIR}/tpl/openmpi/lib:${SCRIPT_DIR}/tpl/ucx/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"

#===========================================
# GPU Configurations
# Format: "NUM_GPUS P Q"
#===========================================
declare -a GPU_CONFIGS=(
    "1 1 1"
    "2 2 1"
    "4 2 2"
    "8 2 4"
)

#===========================================
# Helper Functions
#===========================================
print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

print_separator() {
    echo "----------------------------------------------"
}

# Calculate N for a given number of GPUs
# N scales as sqrt(num_gpus) for weak scaling (constant memory per GPU)
# Result is rounded down to nearest multiple of NB
calculate_N() {
    local num_gpus=$1
    local base_n=$2
    local nb=$3
    
    # N scales as sqrt(num_gpus) for 2D distribution
    # Using bc for floating point math
    local scale_factor=$(echo "scale=6; sqrt($num_gpus)" | bc)
    local raw_n=$(echo "scale=0; $base_n * $scale_factor / 1" | bc)
    
    # Round down to nearest multiple of NB
    local n=$(( (raw_n / nb) * nb ))
    
    echo $n
}

run_benchmark() {
    local mode=$1      # "native" or "ozaki"
    local num_gpus=$2
    local P=$3
    local Q=$4
    local N=$5         # Matrix size for this run
    
    local p=$P
    local q=$Q
    
    echo ""
    print_separator
    echo "Running: ${mode} DGEMM, ${num_gpus} GPU(s), P=${P}, Q=${Q}, N=${N}, NB=${NB}"
    print_separator
    
    cd "${BUILD_DIR}"
    
    # MPI arguments
    local mpi_args="--map-by node --rank-by slot --bind-to none"
    
    # Check for UCX
    local ompi_info="${SCRIPT_DIR}/tpl/openmpi/bin/ompi_info"
    if [[ -x "${ompi_info}" ]] && ${ompi_info} 2>/dev/null | grep -q "MCA pml: ucx"; then
        mpi_args="--mca pml ucx --mca btl ^vader,tcp,openib,uct ${mpi_args}"
    fi
    
    local rochpl_args="-P ${P} -Q ${Q} -p ${p} -q ${q} -N ${N} --NB ${NB}"
    
    local start_time=$(date +%s.%N)
    
    if [[ "${mode}" == "ozaki" ]]; then
        # Run with GEMMul8 enabled (GEMMUL8_NUM_MOD_D >= 2)
        ${MPI_BIN} -np ${num_gpus} ${mpi_args} \
            -x GEMMUL8_NUM_MOD_D=${GEMMUL8_NUM_MOD_D} \
            -x GEMMUL8_FASTMODE_D=${GEMMUL8_FASTMODE_D} \
            ${RUN_SCRIPT} ${rochpl_args} 2>&1 | tee "${BUILD_DIR}/last_run_${mode}_${num_gpus}gpu.log"
    else
        # Run native DGEMM (GEMMUL8_NUM_MOD_D=0 disables GEMMul8)
        ${MPI_BIN} -np ${num_gpus} ${mpi_args} \
            -x GEMMUL8_NUM_MOD_D=0 \
            ${RUN_SCRIPT} ${rochpl_args} 2>&1 | tee "${BUILD_DIR}/last_run_${mode}_${num_gpus}gpu.log"
    fi
    
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "${end_time} - ${start_time}" | bc)
    
    # Extract results from HPL.out
    if [[ -f "${BUILD_DIR}/HPL.out" ]]; then
        local result_line=$(grep -E "^WC" "${BUILD_DIR}/HPL.out" | tail -1)
        local passed=$(grep -E "PASSED|FAILED" "${BUILD_DIR}/HPL.out" | tail -1)
        
        echo ""
        echo "Result: ${result_line}"
        echo "Residual: ${passed}"
        echo "Total time: ${elapsed}s"
        
        # Save to results file
        echo "${mode},${num_gpus},${P},${Q},${N},${NB},${result_line},${passed}" >> "${RESULTS_FILE}"
        
        # Backup HPL.out
        cp "${BUILD_DIR}/HPL.out" "${BUILD_DIR}/HPL_${mode}_${num_gpus}gpu_${TIMESTAMP}.out"
    else
        echo "WARNING: HPL.out not found"
        echo "${mode},${num_gpus},${P},${Q},${N},${NB},NO_RESULT,NO_RESULT" >> "${RESULTS_FILE}"
    fi
}

#===========================================
# Pre-flight Checks
#===========================================
print_header "rocHPL Ozaki/GEMMul8 Comparison Benchmark"

echo "Timestamp: ${TIMESTAMP}"
echo "Results file: ${RESULTS_FILE}"
echo ""

# Check prerequisites
if [[ ! -x "${ROCHPL_BIN}" ]]; then
    echo "ERROR: rocHPL binary not found at ${ROCHPL_BIN}"
    exit 1
fi

if [[ ! -x "${MPI_BIN}" ]]; then
    echo "ERROR: MPI binary not found at ${MPI_BIN}"
    exit 1
fi

echo "Configuration:"
echo "  - Base matrix size (1 GPU): N=${N_1GPU}"
echo "  - Block size NB: ${NB}"
echo "  - Matrix sizes scale as N * sqrt(num_gpus)"
echo "  - GEMMul8 moduli (for ozaki): ${GEMMUL8_NUM_MOD_D}"
echo "  - GEMMul8 fast mode: ${GEMMUL8_FASTMODE_D}"
echo "  - rocHPL binary: ${ROCHPL_BIN}"
echo ""
echo "Planned matrix sizes:"
for config in "${GPU_CONFIGS[@]}"; do
    read -r ng p q <<< "${config}"
    n=$(calculate_N ${ng} ${N_1GPU} ${NB})
    mem_gb=$(echo "scale=2; $n * $n * 8 / 1024 / 1024 / 1024" | bc)
    echo "  - ${ng} GPU(s): N=${n} (~${mem_gb} GB total)"
done
echo ""

# Initialize results file
echo "mode,num_gpus,P,Q,N,NB,result_line,residual" > "${RESULTS_FILE}"

#===========================================
# Run Benchmarks
#===========================================
print_header "Starting Benchmarks"

for config in "${GPU_CONFIGS[@]}"; do
    read -r num_gpus P Q <<< "${config}"
    
    # Calculate N for this GPU count (scales with sqrt(num_gpus))
    N=$(calculate_N ${num_gpus} ${N_1GPU} ${NB})
    
    print_header "Testing with ${num_gpus} GPU(s) (P=${P}, Q=${Q}, N=${N})"
    
    # Run native DGEMM first
    run_benchmark "native" "${num_gpus}" "${P}" "${Q}" "${N}"
    
    # Small delay between runs
    sleep 5
    
    # Run Ozaki/GEMMul8
    run_benchmark "ozaki" "${num_gpus}" "${P}" "${Q}" "${N}"
    
    # Delay between GPU configurations
    sleep 5
done

#===========================================
# Summary
#===========================================
print_header "Benchmark Complete"

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Summary:"
echo ""

# Parse and display results
printf "%-8s | %-4s | %-3s | %-3s | %-8s | %-4s | %-15s | %-10s\n" \
    "Mode" "GPUs" "P" "Q" "N" "NB" "GFLOPS" "Status"
echo "---------|------|-----|-----|----------|------|-----------------|------------"

while IFS=',' read -r mode num_gpus P Q N NB result_line residual; do
    if [[ "${mode}" == "mode" ]]; then
        continue  # Skip header
    fi
    
    # Extract GFLOPS from result line (last field)
    gflops=$(echo "${result_line}" | awk '{print $NF}')
    
    # Extract status
    if [[ "${residual}" == *"PASSED"* ]]; then
        status="PASSED"
    elif [[ "${residual}" == *"FAILED"* ]]; then
        status="FAILED"
    else
        status="UNKNOWN"
    fi
    
    printf "%-8s | %-4s | %-3s | %-3s | %-8s | %-4s | %-15s | %-10s\n" \
        "${mode}" "${num_gpus}" "${P}" "${Q}" "${N}" "${NB}" "${gflops}" "${status}"
done < "${RESULTS_FILE}"

echo ""
echo "Individual run logs saved as: last_run_<mode>_<gpus>gpu.log"
echo "HPL output files saved as: HPL_<mode>_<gpus>gpu_${TIMESTAMP}.out"
