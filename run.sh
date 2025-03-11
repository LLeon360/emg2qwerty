#!/bin/bash
# Distributed training script for emg2qwerty

# Default configuration
DEVICES=0
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="transformer_encoder_ctc_small"
EXP_NAME="INVESTIGATE"
BATCH_SIZE=32
CLUSTER="local"
LOG_DIR="logs"
SEED=0

# Training hyperparameters with defaults
LEARNING_RATE=1e-4
MAX_EPOCHS=1000
LOG_EVERY_N_STEPS=50

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Construct the training command
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
  +trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"

# Run the training command and log output
echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"