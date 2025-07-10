#!/bin/bash

# ========== 配置 ==========
CHECKPOINT="runwayml/stable-diffusion-v1-5"
CHECKPOINT_PATH=./model/PA_Final_Model.pth
INPUT_FOLDER=./data/VAL
RESULTS_FOLDER=./results
STEPS=20

PY_SCRIPT=infer.py

# ========== 启动区 ==========
python $PY_SCRIPT \
  --checkpoint $CHECKPOINT \
  --checkpoint_path $CHECKPOINT_PATH \
  --input_folder $INPUT_FOLDER \
  --results_folder $RESULTS_FOLDER \
  --steps $STEPS
