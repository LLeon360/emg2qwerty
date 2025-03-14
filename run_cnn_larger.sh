#!/bin/bash

DEVICES=1
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="tds_conv_ctc_large"
BATCH_SIZE=64
NUM_WORKERS=8
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=1e-4
MAX_EPOCHS=1000
LOG_EVERY_N_STEPS=50

# Optimizer 3
WEIGHT_DECAY=1e-4  # L2 regularization for optimizer

# TDS Conv CTC Large configuration parameters
MLP_FEATURES="[512]"  # Value for Hydra with brackets
MLP_FEATURES_EXP_NAME="512"  # Clean value for experiment name
BLOCK_CHANNELS="[64, 128, 256, 512, 256, 128, 64]"  # Value for Hydra with brackets
BLOCK_CHANNELS_EXP_NAME="7_layer_autoencoder"  # Simplified for experiment name
KERNEL_WIDTH=64

# session splitting parameters
WINDOW_LENGTH=8000  # 4 sec windows for 2kHz EMG

# Build experiment name with module parameters
EXP_NAME="${MODEL}_BS${BATCH_SIZE}_LR${LEARNING_RATE}_SEED${SEED}_EPOCHS${MAX_EPOCHS}_${MLP_FEATURES_EXP_NAME}MLPFeatures_${BLOCK_CHANNELS_EXP_NAME}BlockChannels_${KERNEL_WIDTH}KernelWidth_${WINDOW_LENGTH}Window"
# EXP_NAME="TDS_CONV_LARGE_INVESTIGATE"  # Uncomment to use a simple name instead

mkdir -p ${LOG_DIR}

# Build command with all parameters
CMD="python -m emg2qwerty.train \
  user=\"${USER}\" \
  trainer.accelerator=gpu \
  trainer.devices=${NUM_DEVICES} \
  trainer.num_nodes=${NUM_NODES} \
  +exp_name=\"${EXP_NAME}\" \
  model=\"${MODEL}\" \
  batch_size=${BATCH_SIZE} \
  num_workers=${NUM_WORKERS} \
  seed=${SEED} \
  cluster=${CLUSTER} \
  optimizer.lr=${LEARNING_RATE} \
  trainer.max_epochs=${MAX_EPOCHS} \
  +trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
  module.mlp_features='${MLP_FEATURES}' \
  module.block_channels='${BLOCK_CHANNELS}' \
  module.kernel_width=${KERNEL_WIDTH} \
  datamodule.window_length=${WINDOW_LENGTH} \
  optimizer.weight_decay=${WEIGHT_DECAY}"

echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"