#!/bin/bash

DEVICES=1
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="roformer_encoder_ctc_small"
BATCH_SIZE=32
NUM_WORKERS=8
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=5e-4
MAX_EPOCHS=500
LOG_EVERY_N_STEPS=50

# Model configuration parameters
MLP_FEATURES="[384]"  # actual val for Hydra with brackets
MLP_FEATURES_EXP_NAME="384"  # Clean value for experiment name
NUM_LAYERS=4
D_MODEL=384
NHEAD=6

# session splitting parameters
WINDOW_LENGTH=8000 # 8000 is 4 sec windows for 2kHz EMG

# hop length
HOP_LENGTH=20 # 2kHz / 20 = 100Hz

# Build experiment name with module parameters
EXP_NAME="${MODEL}_BS${BATCH_SIZE}_LR${LEARNING_RATE}_SEED${SEED}_${MLP_FEATURES_EXP_NAME}MLPFeatures_${D_MODEL}DModel_${NUM_LAYERS}Layers_${NHEAD}Heads"
# EXP_NAME="rot_encoder_small"  # Uncomment to use a simple name instead

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
  module.num_layers=${NUM_LAYERS} \
  module.d_model=${D_MODEL} \
  module.nhead=${NHEAD} \
  datamodule.window_length=${WINDOW_LENGTH} \
  logspec.hop_length=${HOP_LENGTH} "

echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"