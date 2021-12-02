import shutil
import os
import glob

for file in glob.glob('slurm-*'):
    shutil.move(file,os.path.join('slurm_outputs',file))