#!/bin/bash

DEVICES=0
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="residual_lstm_ctc"
BATCH_SIZE=64
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=4e-4
WEIGHT_DECAY=2e-4
MAX_EPOCHS=500
LOG_EVERY_N_STEPS=50
GRAD_ACCUM=1

# Current active configuration parameters, med RNN
# Using med RNN as it had 8 train CER, but overfit quite a bit
# TODO: try data aug to prevent overfitting
MLP_FEATURES="[512]"  
MLP_FEAUTRES_EXP_NAME="512"
HIDDEN_SIZE=256
NUM_LAYERS=4
DROPOUT=0.4   # higher dropout
RNN_TYPE="GRU"

# Build experiment name with module parameters
EXP_NAME="${MODEL}_BS${BATCH_SIZE}x${GRAD_ACCUM}_WD${WEIGHT_DECAY}_LR${LEARNING_RATE}_SEED${SEED}_EPOCHS${MAX_EPOCHS}_${MLP_FEAUTRES_EXP_NAME}MLPFeatures_${HIDDEN_SIZE}HiddenSize_${NUM_LAYERS}Layers_${DROPOUT}Dropout_${RNN_TYPE}"
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
  optimizer.weight_decay=${WEIGHT_DECAY} \
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