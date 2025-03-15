#!/bin/bash

DEVICES=0
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="long_seq_rnn_ctc"
BATCH_SIZE=32
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=1e-3  # Using original conv LR
WEIGHT_DECAY=0.005
MAX_EPOCHS=200
LOG_EVERY_N_STEPS=50
GRAD_ACCUM=1

# Current active configuration parameters, med RNN
# Using med RNN as it had 8 train CER, but overfit quite a bit
MLP_FEATURES="[512]"  
MLP_FEAUTRES_EXP_NAME="512"
HIDDEN_SIZE=256
NUM_LAYERS=4
DROPOUT=0.4   # higher dropout
RNN_TYPE="GRU"

# Experiment name - unique for this configuration
EXP_NAME="long_seq_${RNN_TYPE}_BS${BATCH_SIZE}x${GRAD_ACCUM}_WD${WEIGHT_DECAY}_LR${LEARNING_RATE}_SEED${SEED}_EPOCHS${MAX_EPOCHS}_${MLP_FEAUTRES_EXP_NAME}MLPFeatures_${HIDDEN_SIZE}HiddenSize_${NUM_LAYERS}Layers_${DROPOUT}Dropout_20KWindow_40HopLength_withLRSched"

mkdir -p ${LOG_DIR}

# Build command with all parameters
CMD="python -m emg2qwerty.train \
  user=\"${USER}\" \
  trainer.accelerator=gpu \
  trainer.devices=${NUM_DEVICES} \
  trainer.num_nodes=${NUM_NODES} \
  +exp_name=\"${EXP_NAME}\" \
  model=long_seq_rnn_ctc \
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

# Run the command
echo "Starting training with experiment name: ${EXP_NAME}"
echo "${CMD}"
TRAIN_LOG_FILE="${LOG_DIR}/${EXP_NAME}_train_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${TRAIN_LOG_FILE}"

echo "Training complete. Log saved to ${TRAIN_LOG_FILE}"
