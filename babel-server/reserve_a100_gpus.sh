#!/bin/bash
#SBATCH --job-name=a100_gpu_reserve
#SBATCH --output=sbatch_logs/gpu_reserve_%j.out
#SBATCH --error=sbatch_logs/gpu_reserve_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH --mem=400G
# Add your own --mail-user / --mail-type lines if you want SLURM email notifications.

mkdir -p sbatch_logs

COMPUTE_NODE=$(hostname)
echo "=== GPU Reservation Active ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $COMPUTE_NODE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""
echo "SSH in with:  ssh $COMPUTE_NODE"
echo "Cancel with:  scancel $SLURM_JOB_ID"
echo "==============================="

# Keep alive until SLURM kills the job
trap "echo 'Reservation released.'; exit 0" INT TERM
while true; do
    sleep 3600
done
