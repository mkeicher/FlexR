#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <checkpoint_path>"
    exit 1
fi

# Assign arguments to variables
EXPERIMENT_NAME=$1
CHECKPOINT_PATH=$2

# Array of shot values
shots=(1 5 10)

# Loop through the shot values and submit sbatch jobs
for shot in "${shots[@]}"; do
        sbatch python.slurm trainer.py fit --config=FlexR-1_finetune.yaml --data.sampling=${shot}shot128 --model.init_args.clip_checkpoint=${CHECKPOINT_PATH} --trainer.logger.init_args.name=${EXPERIMENT_NAME}_${shot}shot128
done
sbatch python.slurm trainer.py fit --config=FlexR-1_finetune.yaml --data.sampling=fixed128 --model.init_args.clip_checkpoint=${CHECKPOINT_PATH} --trainer.logger.init_args.name=${EXPERIMENT_NAME}_fixed128
sbatch python.slurm trainer.py fit --config=FlexR-1_finetune.yaml --data.sampling=null --model.init_args.clip_checkpoint=${CHECKPOINT_PATH} --trainer.logger.init_args.name=${EXPERIMENT_NAME}_all