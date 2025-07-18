# BiGRU with Multi-head Attention

This repository implements a Bidirectional Gated Recurrent Unit (BiGRU) neural network with Multi-head Attention for sequence classification tasks. Our work demonstrates significant improvements in classification accuracy through the novel combination of bidirectional temporal modeling and multi-head attention mechanisms.

📄 **Research Paper Available**: https://arxiv.org/abs/2506.14830

🎯 **If you find this work useful for your research, please consider citing our paper:**

```bibtex
@article{wen2025bigruAttention,
  title={Optimization of bi-directional gated loop cell based on multi-head attention mechanism for SSD health state classification model},
  author={Wen, Zhizhao and Zhang, Ruoxin and Wang, Chao},
  journal={arXiv preprint arXiv:2506.14830},
  year={2025}
}
```

⭐ **Star this repository** if you find it helpful and **share it** with your research community!

## Model Architecture

The architecture combines bidirectional recurrent neural networks with attention mechanisms for improved sequence modeling:

### Key Components:

1. **Bidirectional GRU**: 
   - Forward GRU processes the input sequence in its original order
   - Backward GRU processes the input sequence in reversed order
   - Both capture temporal dependencies in different directions

2. **Multi-head Attention**:
   - Applied to the concatenated hidden states from both GRUs
   - Enables the model to focus on different parts of the feature representation
   - Captures complex relationships between features

3. **Classification Layers**:
   - Fully connected layer transforms attention outputs
   - Softmax activation for final classification probabilities

## Usage
**Prerequisites: Anaconda**

To set up environment:
```bash
./setup_env.sh
```
To train and evaluate the model:

```bash
conda activate bigru_attention
python -X utf8 main.py
```

Make sure to place your data file (data.xlsx) in the project directory.

## Implementation Details

The implementation uses PyTorch and includes:

- `model.py`: Contains the neural network architecture implementation
- `flip_layer.py`: Custom layer for sequence reversal
- `main.py`: Training and evaluation pipeline

### Model Parameters

- `input_dim`: Dimension of input features
- `hidden_dim`: Hidden dimension for GRU units (default: 5)
- `num_classes`: Number of output classes
- `num_heads`: Number of attention heads for multi-head attention

## Data Processing

The pipeline handles:
- Data normalization to [0,1] range
- Train/test splitting with stratified sampling
- Tensor conversion and batching

## Training and Evaluation

The model is trained using:
- Adam optimizer
- NLL Loss function
- Learning rate scheduling
- Gradient clipping

Evaluation metrics include:
- Classification accuracy
- Confusion matrices
- ROC curves
- Polygon area metrics

## Visualization

The implementation includes functions to visualize:
- Model architecture
- Training progress
- Prediction results
- Confusion matrices
- ROC curves

## Requirements

- Python 3.8+
- PyTorch 1.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- tqdm
- torchviz

## License

This project is licensed under the MIT License - see the LICENSE file for details. 