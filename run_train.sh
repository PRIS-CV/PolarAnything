#!/bin/bash

# ====== 配置 ======
NUM_PROC=1              # 进程数（等于GPU数）
NUM_MACHINES=1          # 机器数
MIXED_PRECISION="no"    # "no"、"fp16"、"bf16"
DYNAMO_BACKEND="no"     # "no" 或 "inductor"

NUM_EPOCHS=10
BATCH_SIZE=4
LR=4e-5
SAVE_CKPT_FREQ=1000
POLARIZATION_DIR=./data/Polarization_Encoding
RGB_DIR=./data/RGB
CHECKPOINT_DIR=./checkpoints
MODEL_DIR=./model

# 训练脚本
PY_SCRIPT=train.py

# ====== 启动区 ======
if [ "$NUM_PROC" -eq 1 ]; then
  echo "启动单卡训练..."
  accelerate launch \
    --num_processes=1 \
    --num_machines=$NUM_MACHINES \
    --mixed_precision=$MIXED_PRECISION \
    --dynamo_backend=$DYNAMO_BACKEND \
    $PY_SCRIPT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_ckpt_freq $SAVE_CKPT_FREQ \
    --polarization_dir $POLARIZATION_DIR \
    --rgb_dir $RGB_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --model_dir $MODEL_DIR
else
  echo "启动多卡训练，进程数：$NUM_PROC ..."
  accelerate launch \
    --num_processes=$NUM_PROC \
    --multi_gpu \
    --num_machines=$NUM_MACHINES \
    --mixed_precision=$MIXED_PRECISION \
    --dynamo_backend=$DYNAMO_BACKEND \
    $PY_SCRIPT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_ckpt_freq $SAVE_CKPT_FREQ \
    --polarization_dir $POLARIZATION_DIR \
    --rgb_dir $RGB_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --model_dir $MODEL_DIR
fi

#   --no-enable_xformers_memory_efficient_attention
