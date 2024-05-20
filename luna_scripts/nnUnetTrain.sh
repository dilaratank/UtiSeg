#!/bin/bash
#
# slurm specific parameters should be defined as comment line starting with #SBATCH
#SBATCH --job-name=nnunetTrain
#SBATCH --gres=gpu:4g.40gb:1   # number of GPUs (type MIG 1g.10gb) 
#SBATCH --partition=luna-long # using luna-short queue for a job that request up to 8h 
#SBATCH --mem=32G              # max memory per node
#SBATCH --cpus-per-task=4      # max CPU cores per process
#SBATCH --time=01-00:00         # time limit (DD-HH:MM)
#SBATCH --nice=100             # allow other priority jobs to go first

set -eu    # Exit immediately on error or on undefined variable

export nnUNet_raw="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/sandbox/dtank/my-scratch/data/nnunet/nnUNet_results"

nnUNetv2_train 3 2d 0 -tr nnUNetTrainer_100epochs --npz
nnUNetv2_train 3 2d 1 -tr nnUNetTrainer_100epochs --npz
nnUNetv2_train 3 2d 2 -tr nnUNetTrainer_100epochs --npz
nnUNetv2_train 3 2d 3 -tr nnUNetTrainer_100epochs --npz
nnUNetv2_train 3 2d 4 -tr nnUNetTrainer_100epochs --npz