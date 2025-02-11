import tensorflow as tf
from typing import Optional, List, Dict, Any
from tensorflow.keras import layers

class FFNHead(tf.keras.layers.Layer):
    def __init__(self, d_model: int, ffn_dim: int, dropout_rate: float):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_dim),
            layers.Activation(tf.nn.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.ffn(x, training=training)

class AblatedTransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ffn_heads: int,
        ffn_dim: int,
        dropout_rate: float
    ):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn_heads = [
            FFNHead(d_model, ffn_dim, dropout_rate)
            for _ in range(num_ffn_heads)
        ]
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(
        self,
        x: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x, x, x, attention_mask=attention_mask)
        x = self.dropout(x, training=training)
        x = residual + x
        
        # FFN with multiple heads
        residual = x
        x = self.layer_norm2(x)
        ffn_outputs = [head(x, training=training) for head in self.ffn_heads]
        x = tf.reduce_mean(tf.stack(ffn_outputs, axis=0), axis=0)
        x = self.dropout(x, training=training)
        x = residual + x
        
        return x

class AblatedTransformer(tf.keras.Model):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 8,
        num_ffn_heads: int = 4,
        ffn_dim: int = 3072,
        dropout_rate: float = 0.1,
        max_position_embeddings: int = 512,
        vocab_size: int = 32000
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.word_embeddings = layers.Embedding(vocab_size, d_model)
        self.position_embeddings = layers.Embedding(max_position_embeddings, d_model)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Transformer layers with ablation
        self.layers = [
            AblatedTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_ffn_heads=num_ffn_heads,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
    
    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        return_hidden_states: bool = False,
        training: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        # Generate position IDs
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        position_ids = tf.broadcast_to(position_ids, tf.shape(input_ids))
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = word_embeds + position_embeds
        
        # Layer norm and dropout
        hidden_states = self.layer_norm(embeddings)
        hidden_states = self.dropout(hidden_states, training=training)
        
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        # Forward pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, training=training)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        if return_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "vocab_size": self.word_embeddings.input_dim,
            "max_position_embeddings": self.position_embeddings.input_dim
        })
        return config 