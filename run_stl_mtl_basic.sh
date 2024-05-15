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

# Base directory for data and outputs
BASE_DIR="."

# Python script path
PYTHON_SCRIPT_PATH="${BASE_DIR}/modules/df_maker.py"
# Loop through each property, preprocess, train, predict, and cleanup
for PROP_NAME in "${PROPERTIES[@]}"; do
  echo "Preprocessing data for $PROP_NAME"
  python $PYTHON_SCRIPT_PATH --property "$PROP_NAME" --base_dir "$BASE_DIR"

  echo "Training model for $PROP_NAME"
  chemprop train \
    --data-path "${BASE_DIR}/data/train_${PROP_NAME}.csv" \
    --task-type regression \
    --output-dir "${BASE_DIR}/chemprop_out/$PROP_NAME" \
    --num-workers 15 \

  echo "Predicting with model for $PROP_NAME"
  chemprop predict \
    --test-path "${BASE_DIR}/data/test_${PROP_NAME}.csv" \
    --model-path "${BASE_DIR}/chemprop_out/$PROP_NAME/model_0/best.pt" \
    --preds-path "${BASE_DIR}/chemprop_out/${PROP_NAME}.csv" \
    --num-workers 15 \

  echo "Cleaning up generated CSV files for $PROP_NAME"
  rm "${BASE_DIR}/data/train_${PROP_NAME}.csv"
  rm "${BASE_DIR}/data/test_${PROP_NAME}.csv"
done

#to run a MTL on all, simply run:
echo "Training multi-task model for all properties"
chemprop train \
  --data-path "${BASE_DIR}/data/train.csv" \
  --task-type regression \
  --output-dir "${BASE_DIR}/chemprop_out/mtl_all" \
  --num-workers 15 \

echo "Predicting with multi-task model for all properties"
chemprop predict \
  --test-path "${BASE_DIR}/data/test.csv" \
  --model-path "${BASE_DIR}/chemprop_out/mtl_all/model_0/best.pt" \
  --preds-path "${BASE_DIR}/chemprop_out/mtl_all.csv" \
  --num-workers 15 \
