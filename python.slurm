#!/bin/sh
 
#SBATCH --job-name=flexr
#SBATCH --output=/home/guests/matthias_keicher/log/slurm/flexr-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/home/guests/matthias_keicher/log/slurm/flexr-%A.err  # Standard error of the script
#SBATCH --time=7-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12 per GPU)
#SBATCH --mem=128G  # Memory in GB (Don't use more than 128 per GPU)
 
source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate flexr
 
srun python $@