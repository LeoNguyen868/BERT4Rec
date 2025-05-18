#!/bin/bash

# Activate conda environment
# Make sure to adjust this if your conda path is different or if you're not using bash
# It's often better to run this script from an already activated environment
# or to source this script rather than executing it directly if you want the environment to persist.
# However, for a self-contained script, this is a common approach.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rec

# Run data generation
python -u gen_data_fin_v1.py \
    --dataset_name="ml-1m" \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --mask_prob=1.0 \
    --dupe_factor=1 \
    --masked_lm_prob=0.2 \
    --prop_sliding_window=0.5 \
    --signature="-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun" \
    --pool_size=10

# Check if data generation was successful (exit code 0)
if [ $? -ne 0 ]; then
    echo "Data generation (gen_data_fin_v1.py) failed. Exiting."
    exit 1
fi

# Run training and evaluation
python -u run_v1.py \
    --train_input_file=./data/ml-1m-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun.train.tfrecord \
    --test_input_file=./data/ml-1m-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun.test.tfrecord \
    --vocab_filename=./data/ml-1m-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun.vocab \
    --user_history_filename=./data/ml-1m-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun.his \
    --checkpointDir=./checkpoints_testrun/ml-1m \
    --signature="-mp1.0-sw0.5-mlp0.2-df1-mpps40-msl200-testrun-64" \
    --do_train \
    --do_eval \
    --bert_config_file=./bert_train/bert_config_ml-1m_64.json \
    --batch_size=32 \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --num_train_steps=10 \
    --num_warmup_steps=1 \
    --learning_rate=1e-4 \
    --save_checkpoints_steps=5 \
    --iterations_per_loop=5 \
    --max_eval_steps=5 | cat

# Check if training/evaluation was successful
if [ $? -ne 0 ]; then
    echo "Training/evaluation (run_v1.py) failed. Exiting."
    exit 1
fi

echo "Script finished successfully." 