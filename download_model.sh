#!/bin/bash
MODEL_NAME=$1
DEST_DIR=${2:-"./models"}

if [[ -z "$MODEL_NAME" ]]; then
  echo "Please provide the Git URL as the first argument."
  exit 1
fi

# Concatenate with the static part
FULL_GIT_URL="https://huggingface.co/$MODEL_NAME"
MODEL_BASE_NAME=$(basename $MODEL_NAME)

echo "=== Downloading the model from HuggingFace ==="
mkdir -p "$DEST_DIR"
git clone "$FULL_GIT_URL" "$DEST_DIR/$MODEL_BASE_NAME"