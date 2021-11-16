#!/bin/bash
#SBATCH --account=PAS2056
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=dist_test_2_nodes

export EXECFILE='train_ddp.py'
export GLOO_SOCKET_IFNAME=eth0

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=12340
export WORLD_SIZE=1

echo job started at `date`
echo on compute node `cat ${PBS_NODEFILE}`

source /users/PAS1906/demilt4/miniconda3/bin/activate
conda activate torch_latest

module load cuda/11.2.2

cd ${SLURM_SUBMIT_DIR}
# /users/PAS1211/osu1053/CSE_5441/lab1
cd ${TMPDIR}

echo job started at `date` >>current.out
time srun python3 ${SLURM_SUBMIT_DIR}/${EXECFILE} --epochs=1 --local_world_size=1 --batch_size=64 --no-cuda >>current.out 2>&1
echo job ended at `date` >>current.out

export SAVEDIR=${SLURM_SUBMIT_DIR}'/tests/resnet_bs_64_cpu.'${PBS_JOBID}
mkdir ${SAVEDIR}
mv current.out ${SAVEDIR}
