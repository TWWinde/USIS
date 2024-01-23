#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=med_usis
#SBATCH --output=med_usis%j.%N.out
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

#conda activate /anaconda3/envs/myenv
pyenv activate venv
module load cuda
# Run your python code

#experiment_1
#python train.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --add_mask  \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG wavelet --channels_G 16  #16

#experiment_2
#python train.py --name usis_wavelet_no_mask --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG wavelet --channels_G 16  #16


#experiment_3.
#python train.py --name usis_oasis_generator --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 2 --add_mask  \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG oasis --channels_G 64

#experiment_4
#python train.py --name usis_oasis_generator_no_mask --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 2  \
#--netDu wavelet  --continue_train \
#--model_supervision 0 --netG oasis --channels_G 64

#experiment_5 unpaired ct autopet 117714
#python train.py --name unpaired_ct_autopet --dataset_mode ct2ctautopet --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/data_nnunet --batch_size 4  \
#--netDu wavelet  --continue_train \
#--model_supervision 0 --netG wavelet --channels_G 16

#experiment_6 no_3d_noise
python train.py --name usis_wavelet_one2one --dataset_mode ct2mri --gpu_ids 0 \
--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --add_mask  \
--netDu wavelet --continue_train --z_dim 0 --no_3dnoise --continue_train \
--model_supervision 0 --netG wavelet --channels_G 16  #16


#test
#python test.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 20 --model_supervision 0  \


