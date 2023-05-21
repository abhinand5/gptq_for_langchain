#!/bin/bash
mkdir -p .tmp
if [ ! -d ".tmp/GPTQ-for-LLaMa" ]; then
    git clone -b cuda https://github.com/abhinand5/GPTQ-for-LLaMa.git .tmp/GPTQ-for-LLaMa
fi
cd .tmp/GPTQ-for-LLaMa && \
    pip3 install -r requirements.txt && \
    cp setup_cuda.py setup.py && \
    python3 -m pip install .