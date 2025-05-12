import torch
import torch.nn as nn

class FlipLayer(nn.Module):
    """
    Custom layer that flips the input tensor along a specified dimension.
    Equivalent to MATLAB's flip(X, 3) function.
    """
    def __init__(self, dim=2):
        super(FlipLayer, self).__init__()
        self.dim = dim 
        
    def forward(self, x):
        # Flips the tensor along the specified dimension
        return torch.flip(x, [self.dim]) 