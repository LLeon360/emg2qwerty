#!/bin/bash

DEVICES=0
NUM_DEVICES=1
USER="single_user"
MODEL="roformer_encoder_ctc_small.yaml"
EXP_NAME="roformer_encoder_ctc_small_test"
CHECKPOINT="'/home/bytemarish/ECEC147/emg2qwerty/logs/2025-03-12/18-07-29/checkpoints/epoch=874-step=52500.ckpt'"
LOG_DIR="logs"

WINDOW_LENGTH=8000 # 8000 is 4 sec windows for 2kHz EMG

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
    checkpoint=\"${CHECKPOINT}\" \
    +exp_name=\"${EXP_NAME}\" \
    datamodule.window_length=8000"

# Run the evaluation command and log output
echo "Running evaluation with checkpoint: ${CHECKPOINT}"
echo "Command: ${CMD}"
EVAL_LOG_FILE="${LOG_DIR}/${EXP_NAME}_eval_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${EVAL_LOG_FILE}"

echo "Evaluation complete. Log saved to ${EVAL_LOG_FILE}"
