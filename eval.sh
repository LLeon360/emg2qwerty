#!/bin/bash

DEVICES=0,1
NUM_DEVICES=2
USER="leonliu360-ucla-org"
MODEL="transformer_encoder_ctc_small"
EXP_NAME="distributed_training"
CHECKPOINT=""
LOG_DIR="logs"

mkdir -p ${LOG_DIR}

if [ -z "${CHECKPOINT}" ]; then
  # Look for the latest log directory matching the experiment name
  LOG_DIRS=$(find ${LOG_DIR} -type d -name "*" | sort -r)
  
  for DIR in ${LOG_DIRS}; do
    # Check if this directory contains our experiment
    if [ -d "${DIR}/checkpoints" ]; then
      # Look for the best checkpoint
      BEST_CKPT=$(find ${DIR}/checkpoints -name "*.ckpt" | grep -v "last.ckpt" | head -n 1)
      if [ ! -z "${BEST_CKPT}" ]; then
        CHECKPOINT=${BEST_CKPT}
        echo "Found checkpoint: ${CHECKPOINT}"
        break
      fi
    fi
  done
  
  if [ -z "${CHECKPOINT}" ]; then
    echo "Error: Could not find a checkpoint for experiment ${EXP_NAME}"
    exit 1
  fi
fi

CMD="python -m emg2qwerty.train \
    user=\"${USER}\" \
    trainer.accelerator=gpu \
    trainer.devices=${NUM_DEVICES} \
    model=\"${MODEL}\" \
    train=False \
    checkpoint=\"${CHECKPOINT}\""

# Run the evaluation command and log output
echo "Running evaluation with checkpoint: ${CHECKPOINT}"
EVAL_LOG_FILE="${LOG_DIR}/${EXP_NAME}_eval_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${EVAL_LOG_FILE}"

echo "Evaluation complete. Log saved to ${EVAL_LOG_FILE}"
