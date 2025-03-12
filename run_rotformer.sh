#!/bin/bash

DEVICES=0
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="roformer_encoder_ctc_small"
BATCH_SIZE=32
NUM_WORKERS=8
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=1e-4
MAX_EPOCHS=1000
LOG_EVERY_N_STEPS=50

EXP_NAME="rot_encoder_small"

mkdir -p ${LOG_DIR}

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
  +trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"

echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"