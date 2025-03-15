#!/bin/bash

# General settings
DEVICES=1
NUM_DEVICES=1
NUM_NODES=1
USER="single_user"
MODEL="transformer_encoder_ctc_2_med"
BATCH_SIZE=32
CLUSTER="local"
LOG_DIR="logs"
SEED=0
LEARNING_RATE=8e-5
WEIGHT_DECAY=1e-4
MAX_EPOCHS=500
LOG_EVERY_N_STEPS=50
GRAD_ACCUM=1

# Transformer architecture settings
MLP_FEATURES="[512]"
MLP_FEATURES_EXP_NAME="512"
D_MODEL=512
NUM_LAYERS=6
NHEAD=8
DROPOUT=0.25

# Build experiment name
EXP_NAME="${MODEL}_BS${BATCH_SIZE}x${GRAD_ACCUM}_WD${WEIGHT_DECAY}_LR${LEARNING_RATE}_SEED${SEED}_EPOCHS${MAX_EPOCHS}_${MLP_FEATURES_EXP_NAME}MLPFeatures_${D_MODEL}DModel_${NUM_LAYERS}Layers_${NHEAD}Heads_${DROPOUT}Dropout"

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
  module.d_model=${D_MODEL} \
  module.num_layers=${NUM_LAYERS} \
  module.nhead=${NHEAD} \
  module.dropout=${DROPOUT} \
  module.causal=True \
  module.pos_embed='rotary'"

echo "${CMD}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
eval "CUDA_VISIBLE_DEVICES=${DEVICES} ${CMD} 2>&1 | tee ${LOG_FILE}"

echo "Training complete. Log saved to ${LOG_FILE}"
echo "To evaluate the trained model, run: ./eval.sh --exp-name ${EXP_NAME}"