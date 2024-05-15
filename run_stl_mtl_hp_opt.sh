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
  chemprop hpopt \
    --data-path "${BASE_DIR}/data/train_${PROP_NAME}.csv" \
    --task-type regression \
    --num-workers 15 \

  # Read the JSON file
  json_file="${BASE_DIR}/chemprop_hpopt/train_$PROP_NAME/best_params.json" \

  # Extract values using jq
  dropout=$(jq '.train_loop_config.dropout' "$json_file")
  ffn_num_layers=$(jq '.train_loop_config.ffn_num_layers' "$json_file")
  ffn_hidden_dim=$(jq '.train_loop_config.ffn_hidden_dim' "$json_file")
  depth=$(jq '.train_loop_config.depth' "$json_file")
  message_hidden_dim=$(jq '.train_loop_config.message_hidden_dim' "$json_file")

  # Print the values (optional)
  echo "dropout: $dropout"
  echo "ffn_num_layers: $ffn_num_layers"
  echo "ffn_hidden_dim: $ffn_hidden_dim"
  echo "depth: $depth"
  echo "message_hidden_dim: $message_hidden_dim"


  echo "Training model for $PROP_NAME"
  chemprop train \
    --data-path "${BASE_DIR}/data/train_${PROP_NAME}.csv" \
    --task-type regression \
    --output-dir "${BASE_DIR}/chemprop_out/$PROP_NAME" \
    --num-workers 15 \
    --dropout $dropout \
    --ffn-num-layers $ffn_num_layers \
    --ffn-hidden-dim $ffn_hidden_dim \
    --depth $depth \
    --message-hidden-dim $message_hidden_dim \

  echo "Predicting with model for $PROP_NAME"
  chemprop predict \
    --test-path "${BASE_DIR}/data/test_${PROP_NAME}.csv" \
    --model-path "${BASE_DIR}/chemprop_out/$PROP_NAME/model_0/best.pt" \
    --preds-path "${BASE_DIR}/chemprop_out/${PROP_NAME}.csv" \
    --num-workers 15 \

  echo "Cleaning up generated CSV files for $PROP_NAME"
  rm "${BASE_DIR}/data/train_${PROP_NAME}.csv"
  rm "${BASE_DIR}/data/test_${PROP_NAME}.csv"
  rm -r "${BASE_DIR}/chemprop_hpopt/train_${PROP_NAME}"

done


chemprop hpopt \
  --data-path "${BASE_DIR}/data/train.csv" \
  --task-type regression \
  --num-workers 15 \
# Read the JSON file
json_file="${BASE_DIR}/chemprop_hpopt/train.json" \

# Extract values using jq
dropout=$(jq '.train_loop_config.dropout' "$json_file")
ffn_num_layers=$(jq '.train_loop_config.ffn_num_layers' "$json_file")
ffn_hidden_dim=$(jq '.train_loop_config.ffn_hidden_dim' "$json_file")
depth=$(jq '.train_loop_config.depth' "$json_file")
message_hidden_dim=$(jq '.train_loop_config.message_hidden_dim' "$json_file")

# Print the values (optional)
echo "dropout: $dropout"
echo "ffn_num_layers: $ffn_num_layers"
echo "ffn_hidden_dim: $ffn_hidden_dim"
echo "depth: $depth"
echo "message_hidden_dim: $message_hidden_dim"


echo "Training model for $PROP_NAME"
chemprop train \
  --data-path "${BASE_DIR}/data/train.csv" \
  --task-type regression \
  --output-dir "${BASE_DIR}/chemprop_out/mtl_all \
  --num-workers 15 \
  --dropout $dropout \
  --ffn-num-layers $ffn_num_layers \
  --ffn-hidden-dim $ffn_hidden_dim \
  --depth $depth \
  --message-hidden-dim $message_hidden_dim \

echo "Predicting with model for mtl_all"
chemprop predict \
  --test-path "${BASE_DIR}/data/test.csv" \
  --model-path "${BASE_DIR}/chemprop_out/mtl_all/model_0/best.pt" \
  --preds-path "${BASE_DIR}/chemprop_out/mtl_all.csv" \
  --num-workers 15 \
