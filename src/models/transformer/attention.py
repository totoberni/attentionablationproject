import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional, Tuple, Dict, Any
import math

class MultiHeadAttention(layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        use_bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = layers.Dense(d_model, use_bias=use_bias)
        self.k_proj = layers.Dense(d_model, use_bias=use_bias)
        self.v_proj = layers.Dense(d_model, use_bias=use_bias)
        self.o_proj = layers.Dense(d_model, use_bias=use_bias)
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        return_attention: bool = False,
        training: bool = False
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        batch_size = tf.shape(query)[0]
        
        # Linear projections and reshape
        q = self._reshape_to_heads(self.q_proj(query))
        k = self._reshape_to_heads(self.k_proj(key))
        v = self._reshape_to_heads(self.v_proj(value))
        
        # Scaled dot-product attention
        d_head = tf.cast(self.d_head, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_head)
        
        if mask is not None:
            scores = tf.where(mask == 0, float('-inf'), scores)
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, v)
        
        # Reshape and project to output
        context = self._reshape_from_heads(context)
        output = self.o_proj(context)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def _reshape_to_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        return tf.transpose(
            tf.reshape(x, [batch_size, seq_len, self.num_heads, self.d_head]),
            [0, 2, 1, 3]
        )
    
    def _reshape_from_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        return tf.reshape(
            tf.transpose(x, [0, 2, 1, 3]),
            [batch_size, seq_len, self.d_model]
        )
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout.rate,
            "use_bias": self.q_proj.use_bias
        })
        return config

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
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        return_attention: bool = False,
        training: bool = False
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        # Create local attention mask
        seq_len = tf.shape(query)[1]
        local_mask = self._create_local_mask(seq_len, self.window_size)
        
        if mask is not None:
            mask = tf.logical_and(mask, local_mask)
        else:
            mask = local_mask
        
        return super().call(
            query,
            key,
            value,
            mask=mask,
            return_attention=return_attention,
            training=training
        )
    
    def _create_local_mask(self, seq_len: int, window_size: int) -> tf.Tensor:
        """Create a mask for local attention."""
        mask = tf.zeros([seq_len, seq_len], dtype=tf.bool)
        for i in range(seq_len):
            start = tf.maximum(0, i - window_size // 2)
            end = tf.minimum(seq_len, i + window_size // 2 + 1)
            mask = tf.tensor_scatter_nd_update(
                mask,
                [[i, j] for j in range(start, end)],
                tf.ones([end - start], dtype=tf.bool)
            )
        return mask
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"window_size": self.window_size})
        return config 