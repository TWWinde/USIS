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
#experiment_1
#python test.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --model_supervision 0 \
#--model_supervision 0 --netG wavelet --channels_G 16 --generate_seg

#experiment_2
python test.py --name usis_wavelet_no_mask --dataset_mode ct2mri --gpu_ids 0 \
--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --model_supervision 0 \
--model_supervision 0 --netG wavelet --channels_G 16  --generate_seg     #16


#experiment_3
#python test.py --name usis_oasis_generator --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 20 --model_supervision 0 \
#--model_supervision 0 --netG oasis --channels_G 64

#experiment_4
#python test.py --name usis_oasis_generator_no_mask --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 20 --model_supervision 0 \
#--model_supervision 0 --netG oasis --channels_G 64