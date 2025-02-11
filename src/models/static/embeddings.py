import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Optional, Tuple, Any
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class PretrainedEmbeddings(layers.Layer):
    def __init__(
        self,
        embedding_path: str,
        embedding_type: str = "glove",
        embedding_dim: int = 300,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        trainable: bool = False
    ):
        super().__init__()
        self.embedding_type = embedding_type.lower()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.trainable = trainable
        
        # Load pretrained embeddings
        self.word2idx, weights = self._load_embeddings(embedding_path)
        
        # Create embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(weights),
            mask_zero=True,
            trainable=trainable
        )
    
    def call(self, input_ids: tf.Tensor) -> tf.Tensor:
        return self.embedding(input_ids)
    
    def _load_embeddings(self, embedding_path: str) -> Tuple[Dict[str, int], np.ndarray]:
        if self.embedding_type == "glove":
            return self._load_glove(embedding_path)
        elif self.embedding_type == "word2vec":
            return self._load_word2vec(embedding_path)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    def _load_glove(self, glove_path: str) -> Tuple[Dict[str, int], np.ndarray]:
        # Convert GloVe to Word2Vec format if needed
        word2vec_output = glove_path + '.word2vec'
        try:
            glove2word2vec(glove_path, word2vec_output)
        except:
            print("Using existing Word2Vec format file")
        
        return self._load_word2vec(word2vec_output)
    
    def _load_word2vec(self, word2vec_path: str) -> Tuple[Dict[str, int], np.ndarray]:
        # Load vectors using gensim
        model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        
        # Initialize embedding matrix
        weights = np.zeros((self.vocab_size, self.embedding_dim))
        word2idx = {}
        
        # Add special tokens
        word2idx['[PAD]'] = self.pad_token_id
        word2idx['[UNK]'] = self.unk_token_id
        weights[self.pad_token_id] = np.zeros(self.embedding_dim)
        weights[self.unk_token_id] = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Add pretrained vectors
        idx = len(word2idx)
        for word in model.index_to_key:
            if idx >= self.vocab_size:
                break
            word2idx[word] = idx
            weights[idx] = model[word]
            idx += 1
        
        return word2idx, weights
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_type": self.embedding_type,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "trainable": self.trainable
        })
        return config

class HybridEmbedding(layers.Layer):
    def __init__(
        self,
        static_embedding: PretrainedEmbeddings,
        contextual_dim: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.static_embedding = static_embedding
        self.embedding_dim = static_embedding.embedding_dim
        self.contextual_dim = contextual_dim
        self.dropout_rate = dropout_rate
        
        # Projection layer for dimension matching
        if self.embedding_dim != contextual_dim:
            self.projection = layers.Dense(contextual_dim)
        else:
            self.projection = None
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, input_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        embeddings = self.static_embedding(input_ids)
        
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        
        return embeddings
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "contextual_dim": self.contextual_dim,
            "dropout_rate": self.dropout_rate
        })
        return config 