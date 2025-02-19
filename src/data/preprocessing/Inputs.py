from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import tensorflow as tf
from datasets import Dataset
from transformers import AutoTokenizer
import spacy
import nltk
from nltk.tokenize import word_tokenize
import logging
from datetime import datetime
import hashlib
import re
from collections import Counter

from src.data.core import (
    ConfigurationManager, 
    ModelManager,
    ModelInput,
    TaskType
)

logger = logging.getLogger(__name__)

class InputProcessor:
    """Handles input processing and tokenization for both static and transformer models."""
    
    def __init__(self, config_manager: ConfigurationManager, model_manager: ModelManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_config = config_manager.get_config('data')
        self.processed_inputs = set()  # Track processed inputs
    
    def process_inputs(
        self,
        texts: List[str],
        dataset_name: str,
        model_type: str = "transformer"
    ) -> ModelInput:
        """Process inputs based on dataset configuration and model type."""
        # Ensure we're working with raw text
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be raw text strings, not tokenized sequences")
            
        dataset_config = self.data_config['datasets'][dataset_name]
        
        # Generate alignment metadata
        input_metadata = {
            'original_text_hashes': [self._hash_text(t) for t in texts],
            'sequence_lengths': [len(t) for t in texts],
            'processing_timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'processed_inputs': list(self.processed_inputs)
        }
        
        # Get appropriate tokenizer from model manager
        tokenizer_config = self.data_config['tokenizers'][model_type]
        tokenizer = self._get_tokenizer(tokenizer_config)
        
        # Clean texts
        cleaned_texts = [self._clean_text(text, model_type) for text in texts]
        
        # Process inputs according to configuration
        encoded = self._tokenize_and_pad(
            cleaned_texts,
            tokenizer,
            dataset_config['max_length'],
            model_type
        )
        
        # Prepare special token masks
        special_token_masks = self._prepare_special_token_masks(
            encoded['input_ids'],
            tokenizer,
            model_type
        )
        
        # Update metadata with tokenization info
        input_metadata.update({
            'tokenizer_info': {
                'name': tokenizer_config['model'],
                'type': tokenizer_config['type'],
                'max_length': dataset_config['max_length'],
                'special_tokens': self._get_special_token_info(tokenizer, model_type)
            },
            'token_maps': encoded.get('word_ids', None),
            'attention_mask': encoded['attention_mask']
        })
        
        # Process task-specific inputs
        task_inputs = {}
        for task in dataset_config['enabled_tasks']:
            if task not in self.processed_inputs:
                task_config = dataset_config['task_overrides'].get(
                    task,
                    self.data_config['tasks'][task]
                )
                task_inputs[task] = self._prepare_task_inputs(
                    task,
                    encoded,
                    tokenizer,
                    task_config,
                    model_type
                )
                self.processed_inputs.add(task)
        
        # Create ModelInput instance
        return ModelInput(
            input_ids=encoded['input_ids'],
            sequence_mask=encoded['attention_mask'],
            cls_token_mask=special_token_masks['cls_mask'],
            sep_token_mask=special_token_masks['sep_mask'],
            decoder_input=encoded.get('decoder_input', None),
            metadata={**input_metadata, 'task_inputs': task_inputs}
        )
    
    def _clean_text(self, text: str, model_type: str) -> str:
        """Clean text based on model type configuration."""
        text = str(text).strip()
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r'([.,!?;:()\[\]{}])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Apply model-specific cleaning
        if model_type == "static":
            tokenizer_config = self.data_config['tokenizers']['static']
            if tokenizer_config.get('lower_case', False):
                text = text.lower()
        
        return text.strip()
    
    def _tokenize_and_pad(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Tokenize and pad texts based on model type."""
        if model_type == "transformer":
            encoded = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='np',
                return_word_ids=True
            )
        else:  # static
            encoded = self._static_tokenize_and_pad(texts, tokenizer, max_length)
        
        return encoded
    
    def _static_tokenize_and_pad(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int
    ) -> Dict[str, np.ndarray]:
        """Tokenize and pad for static embeddings."""
        all_ids = []
        all_masks = []
        word_ids = []
        
        for text in texts:
            tokens = text.split()  # Simple whitespace tokenization
            tokens = ['<cls>'] + tokens + ['<sep>']
            
            # Convert to IDs
            token_ids = [tokenizer.token_to_id(t) for t in tokens]
            word_id_map = list(range(len(tokens)))  # Map tokens to positions
            
            # Truncate if necessary
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                word_id_map = word_id_map[:max_length]
                mask = [1] * max_length
            else:
                # Pad
                padding_length = max_length - len(token_ids)
                mask = [1] * len(token_ids) + [0] * padding_length
                token_ids.extend([tokenizer.pad_token_id] * padding_length)
                word_id_map.extend([-1] * padding_length)
            
            all_ids.append(token_ids)
            all_masks.append(mask)
            word_ids.append(word_id_map)
        
        return {
            'input_ids': np.array(all_ids),
            'attention_mask': np.array(all_masks),
            'word_ids': word_ids
        }
    
    def _prepare_special_token_masks(
        self,
        input_ids: np.ndarray,
        tokenizer: Any,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Create masks for special tokens based on model type."""
        if model_type == "transformer":
            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id
        else:  # static
            cls_id = tokenizer.token_to_id('<cls>')
            sep_id = tokenizer.token_to_id('<sep>')
        
        cls_mask = (input_ids == cls_id).astype(np.int32)
        sep_mask = (input_ids == sep_id).astype(np.int32)
        
        return {'cls_mask': cls_mask, 'sep_mask': sep_mask}
    
    def _get_special_token_info(self, tokenizer: Any, model_type: str) -> Dict[str, int]:
        """Get special token IDs based on model type."""
        if model_type == "transformer":
            return {
                'cls': tokenizer.cls_token_id,
                'sep': tokenizer.sep_token_id,
                'pad': tokenizer.pad_token_id,
                'mask': tokenizer.mask_token_id
            }
        else:  # static
            return {
                'cls': tokenizer.token_to_id('<cls>'),
                'sep': tokenizer.token_to_id('<sep>'),
                'pad': tokenizer.token_to_id('<pad>'),
                'mask': tokenizer.token_to_id('<mask>')
            }
    
    def _prepare_task_inputs(
        self,
        task: str,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare task-specific inputs."""
        if task == "mlm":
            return self._prepare_mlm_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "nsp":
            return self._prepare_nsp_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "lmlm":
            return self._prepare_lmlm_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "ner":
            return self._prepare_ner_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "pos":
            return self._prepare_pos_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "discourse":
            return self._prepare_discourse_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "contrastive":
            return self._prepare_contrastive_inputs(encoded, tokenizer, task_config, model_type)
        else:
            return {}  # Other tasks don't require special input preparation
    
    def _prepare_mlm_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare masked language modeling inputs."""
        input_ids = encoded['input_ids'].copy()
        attention_mask = encoded['attention_mask']
        mask_prob = task_config.get('mask_probability', 0.15)
        
        if model_type == "static":
            # For static embeddings, use word-level masking
            word_ids = encoded.get('word_ids', [])
            word_level_mask = np.random.rand(len(word_ids)) < mask_prob
            # Map word-level mask to token-level for static embeddings
            masked_indices = np.zeros_like(input_ids, dtype=bool)
            for i, word_id in enumerate(word_ids):
                if word_id is not None and word_level_mask[word_id]:
                    masked_indices[i] = True
        else:
            # Transformer implementation with subword handling
            probability_matrix = np.random.rand(*input_ids.shape)
            special_tokens = self._get_special_token_info(tokenizer, model_type)
            for special_id in special_tokens.values():
                probability_matrix[input_ids == special_id] = 0
            probability_matrix[attention_mask == 0] = 0
            masked_indices = probability_matrix < mask_prob
            
            indices_random = masked_indices & (np.random.rand(*input_ids.shape) < 0.1)
            indices_keep = masked_indices & ~indices_random & (np.random.rand(*input_ids.shape) < 0.1)
            indices_mask = masked_indices & ~indices_random & ~indices_keep
            
            random_words = np.random.randint(0, tokenizer.vocab_size, size=input_ids.shape)
            input_ids[indices_random] = random_words[indices_random]
            input_ids[indices_mask] = special_tokens['mask']
        
        return {
            'input_ids': input_ids,
            'masked_positions': masked_indices.astype(np.int32)
        }
    
    def _prepare_lmlm_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare local masked language modeling inputs."""
        input_ids = encoded['input_ids'].copy()
        attention_mask = encoded['attention_mask']
        
        min_span = task_config.get('min_span', 2)
        max_span = task_config.get('max_span', 5)
        min_masks = task_config.get('min_masks', 1)
        max_masks = task_config.get('max_masks', 5)
        
        # Generate span masks
        masks = np.zeros_like(input_ids, dtype=bool)
        special_tokens = self._get_special_token_info(tokenizer, model_type)
        
        for i in range(len(input_ids)):
            valid_length = attention_mask[i].sum()
            num_spans = np.random.randint(min_masks, max_masks + 1)
            
            for _ in range(num_spans):
                span_length = np.random.randint(min_span, min(max_span + 1, valid_length))
                start = np.random.randint(0, valid_length - span_length)
                masks[i, start:start + span_length] = True
        
        # Don't mask special tokens
        for special_id in special_tokens.values():
            masks[input_ids == special_id] = False
        
        # For transformers, replace tokens
        if model_type == "transformer":
            input_ids[masks] = special_tokens['mask']
        
        return {
            'input_ids': input_ids,
            'masked_positions': masks.astype(np.int32)
        }
    
    def _prepare_nsp_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare next sentence prediction inputs."""
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        batch_size = len(input_ids)
        
        # Create pairs
        pairs = []
        labels = []
        pair_masks = []
        
        for i in range(0, batch_size - 1, 2):
            if np.random.random() < task_config.get('negative_sampling_ratio', 0.5):
                # Create negative pair
                j = np.random.randint(0, batch_size)
                while j == i or j == i + 1:
                    j = np.random.randint(0, batch_size)
                pairs.append((i, j))
                labels.append(0)
            else:
                # Use consecutive pair
                pairs.append((i, i + 1))
                labels.append(1)
            
            # Create attention mask for pair
            pair_mask = np.concatenate([
                attention_mask[pairs[-1][0]],
                attention_mask[pairs[-1][1]]
            ])
            pair_masks.append(pair_mask)
        
        return {
            'nsp_labels': np.array(labels, dtype=np.int32),
            'pair_masks': np.array(pair_masks, dtype=np.int32)
        }
    
    def _prepare_ner_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare named entity recognition inputs."""
        if model_type == "static":
            # For static embeddings, work with word-level tokens directly
            word_ids = list(range(len(encoded['input_ids'][0])))  # Direct mapping
            alignment_masks = np.ones_like(encoded['input_ids'], dtype=np.int32)
            # Exclude special tokens
            special_tokens = self._get_special_token_info(tokenizer, model_type)
            for special_id in special_tokens.values():
                alignment_masks[encoded['input_ids'] == special_id] = 0
        else:
            # Transformer implementation with subword handling
            word_ids = encoded.get('word_ids', None)
            if not word_ids:
                return {}
            
            alignment_masks = []
            for seq_word_ids in word_ids:
                mask = np.zeros(len(seq_word_ids), dtype=np.int32)
                prev_word_id = None
                for i, word_id in enumerate(seq_word_ids):
                    if word_id != prev_word_id and word_id is not None:
                        mask[i] = 1
                    prev_word_id = word_id
                alignment_masks.append(mask)
            alignment_masks = np.array(alignment_masks)
        
        return {
            'alignment_mask': alignment_masks,
            'word_ids': np.array(word_ids, dtype=np.int32)
        }
        
    def _prepare_pos_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare part-of-speech tagging inputs."""
        # POS tagging uses the same alignment mechanism as NER
        return self._prepare_ner_inputs(encoded, tokenizer, task_config, model_type)
        
    def _prepare_discourse_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare discourse marker prediction inputs."""
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        if model_type == "static":
            # For static embeddings, convert markers to their word-level representations
            marker_tokens = task_config.get('marker_tokens', [])
            marker_ids = [tokenizer.token_to_id(token.lower()) for token in marker_tokens]
        else:
            # Transformer tokenization
            marker_tokens = task_config.get('marker_tokens', [])
            marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in marker_tokens]
        
        # Create marker position mask (works for both architectures)
        marker_masks = np.zeros_like(input_ids, dtype=np.int32)
        for marker_id in marker_ids:
            marker_masks |= (input_ids == marker_id)
        
        # Context window handling (adjusted for architecture)
        window_size = task_config.get('context_window', 5)
        context_masks = np.zeros_like(input_ids, dtype=np.int32)
        
        for i in range(len(marker_masks)):
            marker_positions = np.where(marker_masks[i])[0]
            for pos in marker_positions:
                if model_type == "static":
                    # For static, use word-level windows
                    start = max(0, pos - window_size)
                    end = min(len(input_ids[i]), pos + window_size + 1)
                else:
                    # For transformer, account for subword tokens
                    start = max(0, pos - window_size * 2)  # Larger window for subwords
                    end = min(len(input_ids[i]), pos + window_size * 2 + 1)
                context_masks[i, start:end] = 1
        
        context_masks *= attention_mask
        
        return {
            'marker_positions': marker_masks,
            'context_mask': context_masks
        }
        
    def _prepare_contrastive_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare contrastive learning inputs."""
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        if model_type == "static":
            # For static embeddings, create word-level views
            target_set_masks = []
            for set_idx in range(3):
                # Create word-level masks
                set_mask = np.random.rand(len(input_ids)) < task_config.get(f'set_{set_idx+1}_ratio', 0.3)
                # Expand to token level (same mask for whole word)
                token_mask = np.repeat(set_mask, len(input_ids[0])).reshape(input_ids.shape)
                # Don't mask special tokens or padding
                special_tokens = self._get_special_token_info(tokenizer, model_type)
                for special_id in special_tokens.values():
                    token_mask &= (input_ids != special_id)
                token_mask &= (attention_mask == 1)
                target_set_masks.append(token_mask.astype(np.int32))
        else:
            # Transformer implementation with subword handling
            target_set_masks = []
            for set_idx in range(3):
                set_mask = np.random.rand(*input_ids.shape) < task_config.get(f'set_{set_idx+1}_ratio', 0.3)
                special_tokens = self._get_special_token_info(tokenizer, model_type)
                for special_id in special_tokens.values():
                    set_mask &= (input_ids != special_id)
                set_mask &= (attention_mask == 1)
                target_set_masks.append(set_mask.astype(np.int32))
        
        return {
            'target_set_masks': np.stack(target_set_masks, axis=1),
            'contrastive_attention_mask': attention_mask
        }
    
    def _get_tokenizer(self, config: Dict):
        """Retrieve appropriate tokenizer from model manager."""
        return self.model_manager.get_tokenizer(config['type'], config['model'])
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash of the text for alignment verification."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def verify_alignment(self, input_metadata: Dict, target_metadata: Dict) -> bool:
        """Verify alignment between input and target processing."""
        # Check text hashes match
        input_hashes = input_metadata['original_text_hashes']
        target_hashes = target_metadata['tokenization_info']['original_text_hashes']
        
        if input_hashes != target_hashes:
            logger.error("Text hash mismatch between input and target processing")
            return False
        
        # Check sequence lengths match
        input_lengths = input_metadata['sequence_lengths']
        target_lengths = target_metadata['tokenization_info']['sequence_lengths']
        
        if input_lengths != target_lengths:
            logger.error("Sequence length mismatch between input and target processing")
            return False
        
        # Check token mappings if available
        input_maps = input_metadata.get('token_maps')
        target_maps = target_metadata['tokenization_info'].get('token_maps')
        
        if input_maps and target_maps:
            if not all(len(im) == len(tm) for im, tm in zip(input_maps, target_maps)):
                logger.error("Token mapping length mismatch between input and target processing")
                return False
            
            # Check special token alignment
            input_special_tokens = input_metadata['tokenizer_info']['special_tokens']
            target_special_tokens = target_metadata['tokenization_info']['special_tokens']
            
            if input_special_tokens != target_special_tokens:
                logger.error("Special token mismatch between input and target processing")
                return False
        
        return True
    
    def reconstruct_text(self, input_ids: np.ndarray, metadata: Dict) -> List[str]:
        """Reconstruct original text from input IDs using metadata."""
        tokenizer = self._get_tokenizer(metadata['tokenizer_info'])
        
        # Remove special tokens and padding
        special_tokens = metadata['tokenizer_info']['special_tokens']
        special_token_ids = set(special_tokens.values())
        
        reconstructed_texts = []
        for seq_ids, mask in zip(input_ids, metadata['attention_mask']):
            # Filter out special tokens and padding
            valid_ids = [
                id_ for id_, is_valid in zip(seq_ids, mask)
                if id_ not in special_token_ids and is_valid
            ]
            
            # Decode tokens
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            reconstructed_texts.append(text)
        
        return reconstructed_texts 