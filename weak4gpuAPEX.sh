#!/bin/bash
#SBATCH --account=PAS2056
#SBATCH --cluster=owens
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=weak4apexO1

export EXECFILE='train_ddpApex.py'
export GLOO_SOCKET_IFNAME=eth0

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=12340
export WORLD_SIZE=8

echo job started at `date`
echo on compute node `cat ${PBS_NODEFILE}`

# change to your directory please, not mine
source /users/PAS1906/demilt4/miniconda3/bin/activate
conda activate torch_latest

module load cuda/10.2.89

cd ${SLURM_SUBMIT_DIR}
# /users/PCON0023/hamza23/cse5449/lab2
cd ${TMPDIR}

# tune bucket size with --bc parameter (default 25)
echo job started at `date` >>current.out
time srun python3 ${SLURM_SUBMIT_DIR}/${EXECFILE} --epochs=1 --local_world_size=1 --opt-level O1 >>current.out 2>&1
echo job ended at `date` >>current.out

# I would also change the folder name to match
export SAVEDIR=${SLURM_SUBMIT_DIR}'/tests/resnet18_weak_test_8_gpu_APEX_O1_cpp.'${PBS_JOBID}
mkdir ${SAVEDIR}
mv current.out ${SAVEDIR}
