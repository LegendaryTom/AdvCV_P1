#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=p1-log-%J.out
#SBATCH --job-name=CVP1

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=1
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load anaconda/anaconda3
module load cuda/cuda-10.0

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
echo

#This is the python virtual environment that I found to be closest to the prerequisites for the author's code
source activate pytorch-1.5.1+py38+cuda10.2
#source activate pytorch-1.2.0+cuda10_0+py36
conda list | grep torch

#pip install torchnet --user
#pip install pandas --user


echo $PATH

# Run Program
echo
echo "Starting p1.py"
python p1.py --img_dir='../Data/ImageNet2012/ILSVRC2012_img_val' --batch_size=4

echo
# You’re done!
echo "Ending script..."
date


