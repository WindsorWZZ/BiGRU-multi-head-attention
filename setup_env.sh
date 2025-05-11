#!/bin/bash

# 设置环境名称
ENV_NAME="bigru_attention"

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install conda first."
    exit 1
fi

# 创建新的conda环境
echo "Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.8

# 激活环境
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装依赖
echo "Installing dependencies..."

# 安装PyTorch (CPU版本)
conda install -y pytorch cpuonly -c pytorch

# 安装其他依赖
conda install -y numpy pandas scikit-learn matplotlib tqdm openpyxl

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME"
