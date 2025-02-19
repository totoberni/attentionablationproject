from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer
import spacy
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from src.data.core import (
    ConfigurationManager, 
    ModelManager,
    ModelInput,
    TaskType
)

class InputProcessor:
    """Handles input processing and tokenization for model training."""
    
    def __init__(self, config_manager: ConfigurationManager, model_manager: ModelManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_config = config_manager.get_config('data')
    
    def process_inputs(self, texts: List[str], dataset_name: str) -> ModelInput:
        """Process inputs based on dataset configuration."""
        # Ensure we're working with raw text
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be raw text strings, not tokenized sequences")
            
        dataset_config = self.data_config['datasets'][dataset_name]
        
        # Get appropriate tokenizer from model manager
        tokenizer_config = dataset_config['tokenizer']
        tokenizer = self._get_tokenizer(tokenizer_config)
        
        # Process inputs according to configuration
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=dataset_config['max_length'],
            return_tensors='np'
        )
        
        # Create ModelInput instance
        return ModelInput(
            input_ids=encoded['input_ids'],
            sequence_mask=encoded['attention_mask'],
            cls_token_mask=self._get_special_token_mask(encoded['input_ids'], tokenizer.cls_token_id),
            sep_token_mask=self._get_special_token_mask(encoded['input_ids'], tokenizer.sep_token_id)
        )
    
    def _get_tokenizer(self, config: Dict):
        """Retrieve appropriate tokenizer from model manager."""
        model_type = config['type']
        model_name = config['model']
        return self.model_manager.get_tokenizer(model_type, model_name)
    
    def _get_special_token_mask(self, input_ids: np.ndarray, token_id: int) -> np.ndarray:
        """Create mask for special tokens."""
        return (input_ids == token_id).astype(np.int32)

    def prepare_inputs(
        self,
        texts: Union[str, List[str]],
        task: str
    ) -> Dict[str, tf.Tensor]:
        """Prepare inputs for model training/inference."""
        if isinstance(texts, str):
            texts = [texts]
        
        if task == "mlm":
            return self._prepare_mlm_inputs(texts)
        elif task == "lmlm":
            return self._prepare_lmlm_inputs(texts)
        elif task == "nsp":
            return self._prepare_nsp_inputs(texts)
        else:
            return self._prepare_standard_inputs(texts)
    
    def _prepare_standard_inputs(
        self,
        texts: List[str]
    ) -> Dict[str, tf.Tensor]:
        """Prepare standard inputs with attention masks."""
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def _prepare_mlm_inputs(
        self,
        texts: List[str]
    ) -> Dict[str, tf.Tensor]:
        """Prepare inputs for masked language modeling."""
        # Get standard inputs
        inputs = self._prepare_standard_inputs(texts)
        
        # Apply masking
        mask_prob = self.dataset_config["tasks"]["mlm"]["mask_probability"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Create masking probability matrix
        mask_matrix = tf.random.uniform(tf.shape(input_ids)) < mask_prob
        mask_matrix = tf.logical_and(mask_matrix, attention_mask > 0)
        
        # Create masked input ids
        masked_input_ids = tf.identity(input_ids)
        tokenizer_config = self.config["transformer_tokenizer"]
        mask_token_id = tokenizer_config["special_tokens"]["mask"]
        
        # Apply masking
        masked_input_ids = tf.where(
            mask_matrix,
            tf.fill(tf.shape(input_ids), mask_token_id),
            input_ids
        )
        
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "mask_positions": tf.cast(mask_matrix, tf.int32)
        }
    
    def _prepare_lmlm_inputs(
        self,
        texts: List[str]
    ) -> Dict[str, tf.Tensor]:
        """Prepare inputs for local masked language modeling."""
        inputs = self._prepare_standard_inputs(texts)
        lmlm_config = self.dataset_config["tasks"]["lmlm"]
        
        # Create spans for masking
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = tf.shape(input_ids)[0]
        
        # Generate random spans
        spans = []
        for i in range(batch_size):
            valid_length = tf.reduce_sum(attention_mask[i])
            num_spans = tf.random.uniform(
                [],
                lmlm_config["min_masks"],
                lmlm_config["max_masks"] + 1,
                dtype=tf.int32
            )
            
            batch_spans = []
            for _ in range(num_spans):
                span_length = tf.random.uniform(
                    [],
                    lmlm_config["min_span"],
                    lmlm_config["max_span"] + 1,
                    dtype=tf.int32
                )
                start = tf.random.uniform(
                    [],
                    0,
                    valid_length - span_length,
                    dtype=tf.int32
                )
                batch_spans.append((start, start + span_length))
            spans.append(batch_spans)
        
        # Create masking matrix
        mask_matrix = tf.zeros_like(input_ids, dtype=tf.bool)
        for i, batch_spans in enumerate(spans):
            for start, end in batch_spans:
                indices = tf.range(start, end)
                mask_matrix = tf.tensor_scatter_nd_update(
                    mask_matrix,
                    tf.stack([tf.fill([end-start], i), indices], axis=1),
                    tf.fill([end-start], True)
                )
        
        # Apply masking
        tokenizer_config = self.config["transformer_tokenizer"]
        mask_token_id = tokenizer_config["special_tokens"]["mask"]
        masked_input_ids = tf.where(
            mask_matrix,
            tf.fill(tf.shape(input_ids), mask_token_id),
            input_ids
        )
        
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "mask_positions": tf.cast(mask_matrix, tf.int32)
        }
    
    def _prepare_nsp_inputs(
        self,
        texts: List[str]
    ) -> Dict[str, tf.Tensor]:
        """Prepare inputs for next sentence prediction."""
        nsp_config = self.dataset_config["tasks"]["nsp"]
        batch_size = len(texts) // 2
        
        # Split texts into pairs
        text_pairs = []
        labels = []
        
        for i in range(0, len(texts), 2):
            if i + 1 >= len(texts):
                break
                
            if tf.random.uniform([]) < nsp_config["negative_sampling_ratio"]:
                # Create negative pair
                random_idx = tf.random.uniform(
                    [],
                    0,
                    len(texts),
                    dtype=tf.int32
                )
                text_pairs.append((texts[i], texts[random_idx]))
                labels.append(0)
            else:
                # Use consecutive pair
                text_pairs.append((texts[i], texts[i+1]))
                labels.append(1)
        
        # Tokenize pairs
        inputs = self._prepare_standard_inputs(
            [f"{a} {b}" for a, b in text_pairs]
        )
        
        return {
            **inputs,
            "labels": tf.convert_to_tensor(labels, dtype=tf.int32)
        } 