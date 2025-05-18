#!/bin/bash

# Activate conda environment
# Make sure to adjust this if your conda path is different or if you're not using bash
# It's often better to run this script from an already activated environment
# or to source this script rather than executing it directly if you want the environment to persist.
# However, for a self-contained script, this is a common approach.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rec

# Dataset specific variables for the FULL ml-1m dataset
DATASET_NAME="ml-1m"
SIGNATURE_SUFFIX="-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun"
BERT_CONFIG_FILE="./bert_train/bert_config_ml-1m_64.json"
CHECKPOINT_DIR_SUFFIX="ml-1m"

# Run data generation for the full ml-1m dataset
# USING POOL_SIZE=1 FOR DEBUGGING THE HANG
python -u gen_data_fin_v1.py \
    --dataset_name=${DATASET_NAME} \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --mask_prob=1.0 \
    --dupe_factor=10 \
    --masked_lm_prob=0.2 \
    --prop_sliding_window=0.5 \
    --signature=${SIGNATURE_SUFFIX} \
    --pool_size=1 

# Check if data generation was successful
if [ $? -ne 0 ]; then
    echo "Data generation (gen_data_fin_v1.py) for ${DATASET_NAME} failed. Exiting."
    exit 1
fi

# Run training and evaluation on the full ml-1m dataset
python -u run_v1.py \
    --train_input_file=./data/${DATASET_NAME}${SIGNATURE_SUFFIX}.train.tfrecord \
    --test_input_file=./data/${DATASET_NAME}${SIGNATURE_SUFFIX}.test.tfrecord \
    --vocab_filename=./data/${DATASET_NAME}${SIGNATURE_SUFFIX}.vocab \
    --user_history_filename=./data/${DATASET_NAME}${SIGNATURE_SUFFIX}.his \
    --checkpointDir=./checkpoints_testrun/${CHECKPOINT_DIR_SUFFIX} \
    --signature=${SIGNATURE_SUFFIX}-64 \
    --do_train \
    --do_eval \
    --bert_config_file=${BERT_CONFIG_FILE} \
    --batch_size=32 \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --num_train_steps=10 \
    --num_warmup_steps=1 \
    --learning_rate=1e-4 \
    --save_checkpoints_steps=5 \
    --iterations_per_loop=5 \
    --max_eval_steps=5 | cat

if [ $? -ne 0 ]; then
    echo "Training/evaluation (run_v1.py) for ${DATASET_NAME} failed. Exiting."
    exit 1
fi

echo "Script for ${DATASET_NAME} finished successfully." 