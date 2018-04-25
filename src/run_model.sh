#!/bin/bash
# Usage:
# ./run_model.sh 3 1 vgg 30 ../data/CUBS 2

# This is hard-coded to prevent silly mistakes.
declare -A NUM_OUTPUTS
NUM_OUTPUTS["CUBS"]="200"

GPU_ID=$1
NUM_RUNS=$2
ARCH=$3
NUM_EPOCHS=$4
DATA_ROOT=$5
SCALE=$6
LR=1e-3

for RUN_ID in `seq 1 $NUM_RUNS`;
do
  for DATASET in CUBS; do
    mkdir ../checkpoints/$DATASET
    mkdir ../logs/$DATASET
    LR_DECAY_EVERY=45
    LOG_DIR=../logs/$DATASET/$FT_LAYERS
    mkdir $LOG_DIR
    CKPT_DIR=../checkpoints/$DATASET/$FT_LAYERS
    mkdir $CKPT_DIR
    TAG='racnn_'$ARCH'all_lr'$LR'_lrdecay'$LR_DECAY_EVERY'_'$RUN_ID'_epochs'$NUM_EPOCHS

    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode finetune --arch $ARCH \
      --dataset $DATASET --num_outputs ${NUM_OUTPUTS[$DATASET]} --batch_size 1 \
      --lr $LR  --lr_decay_every $LR_DECAY_EVERY --scale $SCALE\
      --lr_decay_factor 0.1 --finetune_epochs $NUM_EPOCHS \
      --train_path $DATA_ROOT --test_path $DATA_ROOT\
      --save_prefix $CKPT_DIR'/'$TAG'.pt' | tee $LOG_DIR'/'$TAG'.txt'
  done
done
