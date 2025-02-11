import torch
import torch.nn as nn
from typing import List, Optional

class DeepFFNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        layer_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if layer_norm:
                layers.append(nn.LayerNorm(prev_dim))
            
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output projection if needed
        self.output_proj = None
        if hidden_dims[-1] != input_dim:
            self.output_proj = nn.Linear(hidden_dims[-1], input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layers(x)
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states
    
    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

class ResidualFFNetwork(DeepFFNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        layer_norm: bool = True
    ):
        super().__init__(input_dim, hidden_dims, dropout_rate, activation, layer_norm)
        assert hidden_dims[-1] == input_dim, "Output dimension must match input for residual connection"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)

class PositionwiseFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        return residual + x 