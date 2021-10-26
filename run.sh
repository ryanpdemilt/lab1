#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gpus-per-node=1
#SBATCH -A PAS2056
#SBATCH --time 12:0:0

source /users/PAS1906/demilt4/miniconda3/bin/activate
conda activate torch_latest

module load cuda/11.2.2

python3 train.py --batch_size 128