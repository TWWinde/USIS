#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=medical
#SBATCH --output=medical%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --nodes=1
#SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate venv
module load cuda
# module load cuda/11.3


#CUDA_VISIBLE_DEVICES=0 python /misc/no_backups/s1449/USIS/dataloaders/TotalSegmentor.py
#CUDA_VISIBLE_DEVICES=0 python /misc/no_backups/s1449/USIS/dataloaders/TotalSegmentator_combine_masks.py
CUDA_VISIBLE_DEVICES=0 python /misc/no_backups/s1449/USIS/dataloaders/data_preparation_ct_mr.py
# s1449/USIS/dataloaders/TotalSegmentor.py