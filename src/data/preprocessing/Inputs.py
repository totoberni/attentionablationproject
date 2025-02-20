from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import spacy
import nltk
from nltk.tokenize import word_tokenize
import logging
from datetime import datetime
import re
from collections import Counter

from src.data.core import (
    ConfigurationManager,
    ModelManager,
    ModelInput,
    TaskType,
    verify_alignment,
    reconstruct_text,
    hash_text,
    get_special_token_info,
    create_attention_mask,
    get_valid_sequence_mask
)

logger = logging.getLogger(__name__)

class InputProcessor:
    """Handles input processing and tokenization for both static and transformer models."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        model_manager: ModelManager
    ):
        """Initialize the input processor with configuration and model managers."""
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_config = config_manager.get_config('data_config')
        self.processed_inputs = set()
    
    def process_inputs(
        self,
        texts: List[str],
        dataset_name: str,
        model_type: str = "transformer"
    ) -> ModelInput:
        """Process inputs based on dataset configuration and model type."""
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be raw text strings")
        
        dataset_config = self.data_config['datasets'][dataset_name]
        max_length = dataset_config.get('max_length', 512)
        
        # Get tokenizer and process texts
        tokenizer = self._get_tokenizer(model_type, dataset_config)
        cleaned_texts = [self._clean_text(text, model_type) for text in texts]
        
        # Generate metadata
        metadata = {
            'text_hashes': [hash_text(text) for text in texts],
            'sequence_lengths': [len(text.split()) for text in texts],
            'processing_timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'processed_inputs': list(self.processed_inputs)
        }
        
        # Tokenize and encode texts
        encoded = self._tokenize_and_pad(
            cleaned_texts,
            tokenizer,
            max_length,
            model_type
        )
        
        # Create special token masks
        special_token_masks = self._prepare_special_token_masks(
            encoded['input_ids'],
            tokenizer,
            model_type
        )
        
        # Update metadata with tokenization info
        metadata.update({
            'tokenizer_info': {
                'name': dataset_config['tokenizer']['model'],
                'type': dataset_config['tokenizer']['type'],
                'max_length': max_length,
                'special_tokens': get_special_token_info(tokenizer)
            },
            'token_maps': encoded.get('word_ids', None),
            'attention_mask': encoded['attention_mask']
        })
        
        # Process task-specific inputs
        task_inputs = {}
        for task in dataset_config['enabled_tasks']:
            if task not in self.processed_inputs:
                task_config = self._get_task_config(task, dataset_config)
                if task_config:
                    task_inputs[task] = self._prepare_task_inputs(
                        task,
                        encoded,
                        tokenizer,
                        task_config,
                        model_type
                    )
                    self.processed_inputs.add(task)
        
        # Create and return ModelInput instance
        return ModelInput(
            input_ids=encoded['input_ids'],
            sequence_mask=encoded['attention_mask'],
            cls_token_mask=special_token_masks['cls_mask'],
            sep_token_mask=special_token_masks['sep_mask'],
            decoder_input=encoded.get('decoder_input', None),
            metadata={**metadata, 'task_inputs': task_inputs}
        )
    
    def _clean_text(self, text: str, model_type: str) -> str:
        """Clean text based on model type configuration."""
        text = str(text).strip()
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r'([.,!?;:()\[\]{}])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        if model_type == "static":
            tokenizer_config = self.data_config['tokenizers']['static']
            if tokenizer_config.get('lower_case', False):
                text = text.lower()
        
        return text.strip()
    
    def _get_tokenizer(self, model_type: str, dataset_config: Dict) -> Any:
        """Get appropriate tokenizer based on model type and config."""
        tokenizer_config = dataset_config.get('tokenizer', {})
        return self.model_manager.get_tokenizer(
            tokenizer_type=tokenizer_config.get('type', 'wordpiece'),
            model_name=tokenizer_config.get('model', 'bert-base-uncased')
        )
    
    def _get_task_config(self, task: str, dataset_config: Dict) -> Optional[Dict]:
        """Get task-specific configuration."""
        if task not in dataset_config.get('enabled_tasks', []):
            return None
        
        task_config = self.data_config['tasks'].get(task, {})
        task_overrides = dataset_config.get('task_overrides', {}).get(task, {})
        
        return {**task_config, **task_overrides}
    
    def _tokenize_and_pad(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Tokenize and pad texts based on model type."""
        if model_type == "transformer":
            return tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='np',
                return_word_ids=True
            )
        else:
            return self._static_tokenize_and_pad(texts, tokenizer, max_length)
    
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
            tokens = text.split()
            tokens = ['<cls>'] + tokens + ['<sep>']
            
            token_ids = [tokenizer.token_to_id(t) for t in tokens]
            word_id_map = list(range(len(tokens)))
            
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                word_id_map = word_id_map[:max_length]
                mask = [1] * max_length
            else:
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
        """Create masks for special tokens."""
        special_tokens = get_special_token_info(tokenizer)
        
        cls_mask = (input_ids == special_tokens['cls']).astype(np.int32)
        sep_mask = (input_ids == special_tokens['sep']).astype(np.int32)
        
        return {'cls_mask': cls_mask, 'sep_mask': sep_mask}
    
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
        elif task in ["ner", "pos"]:
            return self._prepare_token_level_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "discourse":
            return self._prepare_discourse_inputs(encoded, tokenizer, task_config, model_type)
        elif task == "contrastive":
            return self._prepare_contrastive_inputs(encoded, tokenizer, task_config, model_type)
        else:
            return {}
    
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
        
        special_tokens = get_special_token_info(tokenizer)
        valid_mask = get_valid_sequence_mask(input_ids, list(special_tokens.values()))
        
        probability_matrix = np.random.rand(*input_ids.shape)
        probability_matrix[~valid_mask] = 0
        masked_indices = probability_matrix < mask_prob
        
        # Random word replacement (10% of masked tokens)
        indices_random = masked_indices & (np.random.rand(*input_ids.shape) < 0.1)
        random_words = np.random.randint(0, tokenizer.vocab_size, size=input_ids.shape)
        input_ids[indices_random] = random_words[indices_random]
        
        # Masking (90% of masked tokens)
        indices_mask = masked_indices & ~indices_random
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
        
        special_tokens = get_special_token_info(tokenizer)
        valid_mask = get_valid_sequence_mask(input_ids, list(special_tokens.values()))
        
        masks = np.zeros_like(input_ids, dtype=bool)
        for i in range(len(input_ids)):
            valid_length = attention_mask[i].sum()
            num_spans = np.random.randint(min_masks, max_masks + 1)
            
            for _ in range(num_spans):
                span_length = np.random.randint(min_span, min(max_span + 1, valid_length))
                start = np.random.randint(0, valid_length - span_length)
                masks[i, start:start + span_length] = True
        
        masks &= valid_mask
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
        
        pairs = []
        labels = []
        pair_masks = []
        
        for i in range(0, batch_size - 1, 2):
            if np.random.random() < task_config.get('negative_sampling_ratio', 0.5):
                j = np.random.randint(0, batch_size)
                while j == i or j == i + 1:
                    j = np.random.randint(0, batch_size)
                pairs.append((i, j))
                labels.append(0)
            else:
                pairs.append((i, i + 1))
                labels.append(1)
            
            pair_mask = np.concatenate([
                attention_mask[pairs[-1][0]],
                attention_mask[pairs[-1][1]]
            ])
            pair_masks.append(pair_mask)
        
        return {
            'nsp_labels': np.array(labels, dtype=np.int32),
            'pair_masks': np.array(pair_masks, dtype=np.int32)
        }
    
    def _prepare_token_level_inputs(
        self,
        encoded: Dict[str, np.ndarray],
        tokenizer: Any,
        task_config: Dict,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """Prepare token-level task inputs (NER, POS)."""
        if model_type == "static":
            word_ids = list(range(len(encoded['input_ids'][0])))
            alignment_masks = np.ones_like(encoded['input_ids'], dtype=np.int32)
            
            special_tokens = get_special_token_info(tokenizer)
            for special_id in special_tokens.values():
                alignment_masks[encoded['input_ids'] == special_id] = 0
        else:
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
        
        marker_tokens = task_config.get('marker_tokens', [])
        if model_type == "static":
            marker_ids = [tokenizer.token_to_id(token.lower()) for token in marker_tokens]
        else:
            marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in marker_tokens]
        
        marker_masks = np.zeros_like(input_ids, dtype=np.int32)
        for marker_id in marker_ids:
            marker_masks |= (input_ids == marker_id)
        
        window_size = task_config.get('context_window', 5)
        context_masks = np.zeros_like(input_ids, dtype=np.int32)
        
        for i in range(len(marker_masks)):
            marker_positions = np.where(marker_masks[i])[0]
            for pos in marker_positions:
                window_size_adjusted = window_size * (2 if model_type == "transformer" else 1)
                start = max(0, pos - window_size_adjusted)
                end = min(len(input_ids[i]), pos + window_size_adjusted + 1)
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
        special_tokens = get_special_token_info(tokenizer)
        
        target_set_masks = []
        for set_idx in range(3):
            ratio = task_config.get(f'set_{set_idx+1}_ratio', 0.3)
            if model_type == "static":
                set_mask = np.random.rand(len(input_ids)) < ratio
                token_mask = np.repeat(set_mask, len(input_ids[0])).reshape(input_ids.shape)
            else:
                token_mask = np.random.rand(*input_ids.shape) < ratio
            
            valid_mask = get_valid_sequence_mask(input_ids, list(special_tokens.values()))
            token_mask &= valid_mask & (attention_mask == 1)
            target_set_masks.append(token_mask.astype(np.int32))
        
        return {
            'target_set_masks': np.stack(target_set_masks, axis=1),
            'contrastive_attention_mask': attention_mask
        } 