import torch
import torch.nn as nn
import torch.nn.functional as F
from flip_layer import FlipLayer

class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer implementation similar to MATLAB's selfAttentionLayer
    """
    def __init__(self, input_dim, output_dim, num_heads=1):
        super(SelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        print(f"input_dim: {input_dim}, output_dim: {output_dim}, num_heads: {num_heads}")

        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Projection to output dimension if needed
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        
        # Ensure x has sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Apply multi-head attention
        attn_output, _ = self.multihead_attention(x, x, x)
        
        # Apply projection if needed
        output = self.projection(attn_output)
        
        # Return the output with same dimensions as input
        if output.size(1) == 1:
            output = output.squeeze(1)
            
        return output

class BiGRUAttentionModel(nn.Module):
    """
    Bidirectional GRU with Multi-head Attention model, 
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads):
        super(BiGRUAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Forward GRU
        self.gru_forward = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Flip layer
        self.flip = FlipLayer()  # Flip along sequence dimension
        
        # Backward GRU
        self.gru_backward = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Self-attention layer
        self.self_attention = SelfAttentionLayer(
            input_dim=2*hidden_dim,
            output_dim=2*hidden_dim,
            num_heads=num_heads
        )
        
        # Fully connected layer
        self.fc = nn.Linear(2*hidden_dim, num_classes)
        
        # Softmax layer
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        # X shape: (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        
        # Ensure x has sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension (batch_size, 1, input_dim)
            
        # Forward GRU
        _, forward_hidden = self.gru_forward(x)
        
        # Get the last hidden state
        forward_hidden = forward_hidden.squeeze(0)  # (batch_size, hidden_dim)
        
        # Flip the input sequence and process through backward GRU
        x_flipped = self.flip(x)
        _, backward_hidden = self.gru_backward(x_flipped)
        
        # Get the last hidden state
        backward_hidden = backward_hidden.squeeze(0)  # (batch_size, hidden_dim)
        
        # Concatenate the forward and backward hidden states
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # (batch_size, 2*hidden_dim)
        
        # Apply self-attention on the input
        attention_out = self.self_attention(concat_hidden)
        
        # Apply fully connected layer to the concatenated hidden states
        fc_out = self.fc(attention_out)

        # # Apply softmax to the output
        output = self.soft(fc_out)
        
        return output 