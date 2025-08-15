#!/bin/sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH="..:."

export PYTHONHASHSEED=0
export STARTING_PERCENTAGE=0
export GAME=zork1
export SEED=1
export ALPHA=0.15

echo "USING GPU ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"


python train.py \
  --game_folder_path ../annotated_games/${GAME} \
  --lm_path model_weights/gpt2 \
  --output_dir ./logs/loss_ft/${GAME}_start${STARTING_PERCENTAGE}_ft\
  --seed 1 \
  --log_freq 100 \
  --num_envs 8 \
  --batch_size 64 \
  --lm_top_k 40 \
  --max_steps 50000 \
  --starting_percentage ${STARTING_PERCENTAGE} \
  --alpha ${ALPHA}


