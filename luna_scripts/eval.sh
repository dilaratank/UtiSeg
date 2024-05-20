#!/bin/bash
#
# slurm specific parameters should be defined as comment line starting with #SBATCH
#SBATCH --job-name=eval
#SBATCH --gres=gpu:4g.40gb:1   # number of GPUs (type MIG 1g.10gb) 
#SBATCH --partition=luna-short # using luna-short queue for a job that request up to 8h 
#SBATCH --mem=32G              # max memory per node
#SBATCH --cpus-per-task=4      # max CPU cores per process
#SBATCH --time=0-01:00         # time limit (DD-HH:MM)
#SBATCH --nice=100             # allow other priority jobs to go first

set -eu    # Exit immediately on error or on undefined variable

# some applications can use a specific number of cores, so specify how many are reserved
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/sandbox/dtank/MScUtiSeg/
source utiseg/bin/activate

wandb login 1f7ca4c3e42dd13df4cd9dd42a5b342986bad2c1

python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-fold2-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-5wjfvkly:v0" --f 2
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-fold3-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-q4e5col1:v0" --f 3
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-fold4-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-gmnnvhhl:v0" --f 4
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-fold5-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-c9yjnqrm:v0" --f 5

python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-randomrot-fold2-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-qegao9z6:v0" --f 2
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-randomrot-fold3-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-vfy6se1k:v0" --f 3
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-randomrot-fold4-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-f5e3dw6s:v0" --f 4
python eval.py --imaging_type "ALL" --img_size 256 --run_name "ALL-8-256-randomrot-fold5-round2-EVAL" --model_path "dilaratank/MScUtiSeg/model-li7020b7:v0" --f 5
