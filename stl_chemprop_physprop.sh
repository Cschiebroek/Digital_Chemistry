#!/bin/bash

# Define the list of property names
PROPERTIES=(
  "LogVP"
  "LogP"
  "LogOH"
  "LogBCF"
  "LogHalfLife"
  "BP"
  "Clint"
  "FU"
  "LogHL"
  "LogKmHL"
  "LogKOA"
  "LogKOC"
  "MP"
  "LogMolar"
)

# Python script path
PYTHON_SCRIPT_PATH="/localhome/cschiebroek/other/Digital_Chemistry/df_maker.py"

# Base directory for data and outputs
BASE_DIR="/localhome/cschiebroek/other/Digital_Chemistry"

# Loop through each property, preprocess, train, predict, and cleanup
for PROP_NAME in "${PROPERTIES[@]}"; do
  echo "Preprocessing data for $PROP_NAME"
  python $PYTHON_SCRIPT_PATH --property "$PROP_NAME"

  echo "Training model for $PROP_NAME"
  chemprop train \
    --data-path "${BASE_DIR}/train_${PROP_NAME}.csv" \
    --task-type regression \
    --output-dir "${BASE_DIR}/chemprop_out/$PROP_NAME" \
    --num-workers 15 \

  echo "Predicting with model for $PROP_NAME"
  chemprop predict \
    --test-path "${BASE_DIR}/test_${PROP_NAME}.csv" \
    --model-path "${BASE_DIR}/chemprop_out/$PROP_NAME/model_0/best.pt" \
    --preds-path "${BASE_DIR}/chemprop_out/${PROP_NAME}.csv" \
    --num-workers 15 \

  echo "Cleaning up generated CSV files for $PROP_NAME"
  rm "${BASE_DIR}/train_${PROP_NAME}.csv"
  rm "${BASE_DIR}/test_${PROP_NAME}.csv"
done

#to run a MTL on all, simply run:
# chemprop train --data-path /path/to/train.csv --task-type regression --output-dir /path/to/output_dir --num-workers 15
#and
#chemprop predict --test-path path/to/test.csv --model-path /path/to/model/model_0/best.pt --preds-path /path/to/output_dir/preds_all_tasks_chemprop.csv