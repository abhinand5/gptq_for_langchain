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

echo "=== Making GPTQ-for-LLaMa importable ==="
# Rename GPTQ dir so that it is importable
if [ -d ".tmp/GPTQ-for-LLaMa" ]; then
    mv .tmp/GPTQ-for-LLaMa .tmp/gptq_for_llama
fi
# Add the init file so that Python treats it as a package
if [ ! -f ".tmp/gptq_for_llama/__init__.py" ]; then
    touch .tmp/gptq_for_llama/__init__.py
fi

# Append the GPTQ directory to the PYTHONPATH
FULL_PATH=$(readlink -f ".tmp")
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "#!/bin/sh" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "#!/bin/sh" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo export "PYTHONPATH=$FULL_PATH:$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo export "PYTHONPATH=$PYTHONPATH:$FULL_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

echo "DONE"