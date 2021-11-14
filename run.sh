#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -A PAS2056
#SBATCH --time 03:00:00

source /users/PAS1906/demilt4/miniconda3/bin/activate
conda activate torch_latest

module load cuda/11.2.2

python3 /users/PAS1906/demilt4/miniconda3/envs/torch_latest/lib/python3.8/site-packages/torch/distributed/launch.py --nnode=2 --node_rank=0 --nproc_per_node=1 train_ddp.py --epochs 5 --local_world_size=1
python3 -m torch.distributed.run --nnode=2 --node_rank=0 --nproc_per_node=1 train_ddp.py --epochs 5 --local_world_size=1