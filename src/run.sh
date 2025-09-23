#!/bin/bash

# ==============================
# run.sh - EEG ML pipeline runner
# ==============================

# Exit immediately if a command exits with a non-zero status
set -e

# Default parameters
DATA_DIR="./data"
SAVE_DIR="./results"
KFOLDS=10
BAND="ci95"
PLOT_MODE="both"

echo "===================================="
echo " Running EEG ML Pipeline"
echo " Data dir: $DATA_DIR"
echo " Save dir: $SAVE_DIR"
echo " K-folds:  $KFOLDS"
echo " Band:     $BAND"
echo " Plot:     $PLOT_MODE"
echo "===================================="

# Make output dir if it doesn’t exist
mkdir -p $SAVE_DIR

# Call Python script
python roc.py \
    --plot $PLOT_MODE \
    --kf $KFOLDS \
    --band $BAND \
    --data_dir $DATA_DIR \
    --savedir $SAVE_DIR

echo "===================================="
echo "✅ Done! Results saved in $SAVE_DIR"
echo "===================================="
