#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --mail-user=yjiang5@wpi.edu
#SBATCH --mail-type=ALL

module load python/3.7.13/jz4yxoc
module load cuda11.6/toolkit/11.6.2
module load matlab/R2020a 

source /home/yjiang5/ConformalAblation/venv/bin/activate

echo $SLURM_JOB_NODELIST
echo $PWD

# python3 main.py --test_dqn
python3 main.py