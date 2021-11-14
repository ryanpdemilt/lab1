#!/bin/bash
#SBATCH --account=PAS2056
#SBATCH --cluster=pitzer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=01:00:00
#SBATCH --job-name=imdb_wiki_bs_8

export EXECFILE='train.py'
echo job started at `date`
echo on compute node `cat ${PBS_NODEFILE}`

source /users/PAS1906/demilt4/miniconda3/bin/activate
conda activate torch_latest

module load cuda/11.2.2

cd ${SLURM_SUBMIT_DIR}
# /users/PAS1211/osu1053/CSE_5441/lab1
cd ${TMPDIR}

echo job started at `date` >>current.out
time python3 ${SLURM_SUBMIT_DIR}/${EXECFILE} --batch_size 16 --epochs 1 >>current.out 2>&1
echo job ended at `date` >>current.out

export SAVEDIR=${SLURM_SUBMIT_DIR}'/tests/batch_size_8.'${PBS_JOBID}
mkdir ${SAVEDIR}
mv current.out ${SAVEDIR}