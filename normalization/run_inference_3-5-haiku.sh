#!/bin/bash

#SBATCH -c 1              # Number of cores
#SBATCH -t 2-00:00:00            # Runtime in D-HH:MM:SS (48 hours)
#SBATCH -p jsteinhardt          # Partition to submit to
#SBATCH --mem=10000           # Memory in MB
#SBATCH -o ./results/HCM-3k/exp_4/%j.out  # File to which STDOUT will be written
#SBATCH -e ./results/HCM-3k/exp_4/%j.err   # File to which STDERR will be written
#SBATCH --job-name=inference_3-5-haiku  # Job name

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to the project root directory
# Use SLURM_SUBMIT_DIR if available (set by SLURM), otherwise use script location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    # Fallback: get script's actual location and go to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi

# Run the inference script
python normalization/inference_w_anthropic.py \
    --input_path ./results/HCM-3k/legacy/responses_gpt-4.1-mini.json \
    --output_path ./results/HCM-3k/exp_4/responses_claude-4-5-haiku.json \
    --model claude-haiku-4-5-20251001 \
    --anthropic_api_key "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" \
    --num_runs 5 \
    --temperature 0.7

# Print completion information
echo ""
echo "End Time: $(date)"
echo "Job completed successfully"
