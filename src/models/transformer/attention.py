import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self._reshape_to_heads(self.q_proj(query))
        k = self._reshape_to_heads(self.k_proj(key))
        v = self._reshape_to_heads(self.v_proj(value))
        
        # Scaled dot-product attention
        d_head = self.d_head
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and project to output
        context = self._reshape_from_heads(context)
        output = self.o_proj(context)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    
    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_head = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

class LocalAttention(MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 128,
        dropout_rate: float = 0.1
    ):
        super().__init__(d_model, num_heads, dropout_rate)
        self.window_size = window_size
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Create local attention mask
        seq_len = query.size(1)
        local_mask = self._create_local_mask(seq_len, self.window_size, query.device)
        
        if mask is not None:
            mask = mask & local_mask
        else:
            mask = local_mask
        
        return super().forward(query, key, value, mask, return_attention)
    
    @staticmethod
    def _create_local_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.bool() 