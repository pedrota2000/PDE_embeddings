#!/bin/bash
#SBATCH --job-name=wavelets
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --constraint=h100  # if you want a particular type of GPU
#SBATCH --time=6-1
#SBATCH --mail-type=ALL       # Send email on job start, end, and failure
#SBATCH --mail-user=pedro.tarancon@icc.ub.edu  # Replace with your email address
module purge
source /mnt/home/ptarancon/python_env/neurodiffeq/bin/activate
# Run the first PyTorch training script
python MH_Burgers_wavelets.py --gpu 0

# Run the second PyTorch training script
#python3 train_2.py --gpu 0 &
#python3 train_3.py --gpu 0 &
#python3 train_4.py --gpu 0 &
#python3 train_5.py --gpu 0 &
