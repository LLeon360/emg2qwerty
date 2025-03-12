#!/bin/bash

# Basic training configuration
DEVICES=1
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="rnn_ctc"
BATCH_SIZE=64
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=4e-4
MAX_EPOCHS=1000
LOG_EVERY_N_STEPS=50
GRAD_ACCUM=1
# NOTE: gradient accumulation is defined manually in residual_rnn_ctc.yaml and rnn_ctc.yaml

# Alternate configuration parameters (from commented-out section)
MLP_FEATURES="[512]"  # Value for Hydra with brackets
MLP_FEATURES_EXP_NAME="512"  # Clean value for experiment name
HIDDEN_SIZE=512
NUM_LAYERS=4
DROPOUT=0.3
RNN_TYPE="GRU"

# Build experiment name with module parameters
EXP_NAME="${MODEL}_BS${BATCH_SIZE}x${GRAD_ACCUM}_LR${LEARNING_RATE}_SEED${SEED}_EPOCHS${MAX_EPOCHS}_${MLP_FEATURES_EXP_NAME}MLPFeatures_${HIDDEN_SIZE}HiddenSize_${NUM_LAYERS}Layers_${DROPOUT}Dropout_${RNN_TYPE}"
# EXP_NAME="RNN_CTC_INVESTIGATE"

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
  seed=${SEED} \
  cluster=${CLUSTER} \
  optimizer.lr=${LEARNING_RATE} \
  trainer.max_epochs=${MAX_EPOCHS} \
  +trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
  module.mlp_features='${MLP_FEATURES}' \
  module.rnn_hidden_size=${HIDDEN_SIZE} \
  module.rnn_num_layers=${NUM_LAYERS} \
  module.dropout=${DROPOUT} \
  module.rnn_type=\"${RNN_TYPE}\""

echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"