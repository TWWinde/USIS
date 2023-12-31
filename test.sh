#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=med_test
#SBATCH --output=med_test%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --nodes=1
# SBATCH --gpus=rtx_a5000:1
# SBATCH --gpus=geforce_rtx_2080ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1

# Activate everything you need

#conda activate /anaconda3/envs/myenv
pyenv activate venv
module load cuda
# Run your python code

#test
python test.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
--dataroot /misc/data/private/autoPET/CT_MR --batch_size 20 --model_supervision 0