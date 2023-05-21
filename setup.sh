#!/bin/bash
echo "=== Installing PyTorch, CUDA and other key requirements ==="
conda install -y -k pytorch[version=2,build=py3.10_cuda11.7*] torchvision torchaudio pytorch-cuda=11.7 cuda-toolkit ninja git -c pytorch -c nvidia/label/cuda-11.7.0 -c nvidia

echo "=== Setting up Quant CUDA package ==="
bash ./setup_quant.sh

echo "=== Downloading the WizardLM 7B GPTQ Model from HF ==="
mkdir -p ./models
git clone https://huggingface.co/TheBloke/wizardLM-7B-GPTQ ./models/wizardLM-7B-GPTQ

echo "=== Installing Required packages ==="
pip install langchain==0.0.175 \
    transformers==4.28.0 \
    sentence-transformers==2.2.2 \
    accelerate==0.18.0 \
    bitsandbytes==0.38.1 \
    safetensors==0.3.0