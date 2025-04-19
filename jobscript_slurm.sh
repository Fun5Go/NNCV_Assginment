#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH --time=10:30:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh