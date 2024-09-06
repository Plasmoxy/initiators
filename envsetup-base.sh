# Usage: source ./envsetup-base.sh <env_name>
#!/bin/bash
set -e

# Setup environment
mamba create -y --name $1 python=3.10
mamba activate $1

# install base for pytorch
mamba install -y pytorch-cuda=12.1 pytorch cudatoolkit-dev xformers -c pytorch -c nvidia -c xformers

# Latest core
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git

# Proto and spiece deps
pip install protobuf sentencepiece

# Fill-in other deps
cuda_major_version=$(python -c "import torch; print(torch.cuda.get_device_capability()[0], end='')")
if [ $cuda_major_version -ge 8 ]; then
    # Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)
    pip install --no-deps packaging ninja einops flash-attn xformers trl
else
    # (V100, Tesla T4, RTX 20xx)
    pip install --no-deps xformers trl peft accelerate bitsandbytes
fi

# GPTQ
BUILD_CUDA_EXT=0 pip install auto-gptq

# Jupyter and other tools
pip install jupyterlab ipywidgets papermill wandb datasets sacremoses tiktoken

# Jupyter kernel install
python -m ipykernel install --user --name $1
