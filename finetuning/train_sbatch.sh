#!/bin/bash
#SBATCH --job-name=prm_sft_train
#SBATCH --output=sbatch_logs/prm_sft_train_%j.out
#SBATCH --error=sbatch_logs/prm_sft_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-q9-28,babel-q5-24"

mkdir -p sbatch_logs

echo "=== PRM SFT Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the training script
bash /home/srgandhi/tool-overuse/finetuning/run_qwen3_8b_prm_full_sft_l40s.sh
