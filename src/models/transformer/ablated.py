import torch
import torch.nn as nn
from typing import Optional

class AblatedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 8,
        num_ffn_heads: int = 4,
        ffn_dim: int = 3072,
        dropout_rate: float = 0.1,
        max_position_embeddings: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.word_embeddings = nn.Embedding(32000, d_model)  # vocab size from config
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer layers with ablation
        self.layers = nn.ModuleList([
            AblatedTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_ffn_heads=num_ffn_heads,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ):
        # Generate position IDs
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = word_embeds + position_embeds
        
        # Layer norm and dropout
        hidden_states = self.layer_norm(embeddings)
        hidden_states = self.dropout(hidden_states)
        
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        # Forward pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        if return_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states

class AblatedTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ffn_heads: int,
        ffn_dim: int,
        dropout_rate: float
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.ffn_heads = nn.ModuleList([
            FFNHead(d_model, ffn_dim, dropout_rate)
            for _ in range(num_ffn_heads)
        ])
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x, x, x, attn_mask=attention_mask)[0]
        x = self.dropout(x)
        x = residual + x
        
        # FFN with multiple heads
        residual = x
        x = self.layer_norm2(x)
        ffn_output = torch.stack([head(x) for head in self.ffn_heads])
        x = ffn_output.mean(dim=0)  # Average across FFN heads
        x = self.dropout(x)
        x = residual + x
        
        return x

class FFNHead(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout_rate: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) 