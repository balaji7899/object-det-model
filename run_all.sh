#!/bin/bash

# run_all.sh
# This script orchestrates the execution of the ACADA project:
# 1. Target synthesis (preprocessing)
# 2. Training
# 3. Evaluation
#
# Ensure that you have installed the required packages:
#   pip install -r requirements.txt
#
# Make sure to update directory paths if needed.

# Stop the script if any command fails.
set -e

echo "===================================="
echo "Starting the ACADA project workflow"
echo "===================================="

# 1. Preprocessing: Synthesize target domain images
echo "Step 1: Synthesizing target domain images..."
python preprocessing/target_synthesis.py \
  --source_train data/source/train \
  --source_val data/source/val \
  --output_foggy_train data/target/foggy/train \
  --output_foggy_val data/target/foggy/val \
  --output_lowlight_train data/target/lowlight/train \
  --output_lowlight_val data/target/lowlight/val \
  --output_artistic_train data/target/artistic/train \
  --output_artistic_val data/target/artistic/val \
  --fog_intensity 0.5 \
  --lowlight_intensity 0.5 \
  --artistic_intensity 0.5 \
  --split_ratio 0.8

echo "Target domain synthesis complete."
echo "===================================="

# 2. Training: Run the training pipeline
echo "Step 2: Starting model training..."
python train.py \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 0.001 \
  --lambda_adv 1.0 \
  --lambda_con 0.1 \
  --lambda_det 1.0 \
  --temperature 0.07 \
  --source_train data/source/train \
  --source_val data/source/val \
  --target_foggy data/target/foggy \
  --target_lowlight data/target/lowlight \
  --target_artistic data/target/artistic \
  --checkpoint_dir checkpoints \
  --log_dir logs

echo "Training complete. Checkpoints and logs saved in 'checkpoints' and 'logs'."
echo "===================================="

# 3. Evaluation: Evaluate the model on source and target validation sets
echo "Step 3: Evaluating the model..."
python evaluation.py \
  --checkpoint_path checkpoints/best_model.pth \
  --batch_size 8 \
  --source_val data/source/val \
  --target_foggy_val data/target/foggy/val \
  --target_lowlight_val data/target/lowlight/val \
  --target_artistic_val data/target/artistic/val \
  --log_dir logs_evaluation

echo "Evaluation complete. Check evaluation logs and visuals in 'logs_evaluation' and 'evaluation_visuals'."
echo "===================================="
echo "ACADA project workflow finished successfully."
