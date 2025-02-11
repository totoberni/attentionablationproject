import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Optional, Dict, Any

class DeepFFNetwork(tf.keras.Model):
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
        self.layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if layer_norm:
                self.layers_list.append(layers.LayerNormalization(epsilon=1e-6))
            
            self.layers_list.extend([
                layers.Dense(hidden_dim),
                self._get_activation(activation),
                layers.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output projection if needed
        if hidden_dims[-1] != input_dim:
            self.output_proj = layers.Dense(input_dim)
        else:
            self.output_proj = None
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x
    
    @staticmethod
    def _get_activation(activation: str) -> layers.Layer:
        if activation.lower() == "gelu":
            return layers.Activation(tf.nn.gelu)
        elif activation.lower() == "relu":
            return layers.ReLU()
        elif activation.lower() == "tanh":
            return layers.Activation("tanh")
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims
        })
        return config

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
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return x + super().call(x, training=training)

class PositionwiseFFN(layers.Layer):
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
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff),
            layers.Activation(tf.nn.gelu) if activation.lower() == "gelu" else layers.ReLU(),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.ffn(x, training=training)
        return residual + x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff
        })
        return config 