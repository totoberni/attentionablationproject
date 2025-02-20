from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import tensorflow as tf
import torch
from transformers import pipeline
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score
)
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime

from src.data.core import (
    ConfigurationManager,
    ModelManager,
    TaskType,
    ModelTarget,
    TaskLabels,
    verify_alignment,
    reconstruct_text,
    hash_text,
    get_special_token_info
)

logger = logging.getLogger(__name__)

class TargetGenerator:
    """Handles generation and processing of target labels."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        model_manager: ModelManager
    ):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.config = config_manager.get_config("data_config")
    
    def _preprocess_text(self, text: str, config: Dict) -> str:
        """Apply preprocessing based on configuration."""
        if config.get("remove_html"):
            text = re.sub(r"<[^>]+>", "", text)
        if config.get("normalize_unicode"):
            text = unicodedata.normalize("NFKC", text)
        if config.get("handle_numbers"):
            text = re.sub(r"\d+", "[NUM]", text)
        return text
    
    def _pad_or_truncate(self, sequence: List[int], max_length: Optional[int] = None) -> List[int]:
        """Pad or truncate sequence to specified length."""
        if max_length is None:
            max_length = self.config.get("max_length", 512)
        if len(sequence) < max_length:
            return sequence + [-100] * (max_length - len(sequence))
        return sequence[:max_length]
    
    def generate_targets(
        self,
        texts: List[str],
        task: TaskType,
        task_config: Dict
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Generate targets based on task type."""
        # Create a copy of texts for preprocessing to avoid modifying originals
        processed_texts = texts.copy()
        
        # Preprocess texts if needed
        if task_config.get("preprocessing", {}):
            processed_texts = [self._preprocess_text(text, task_config["preprocessing"]) for text in processed_texts]
        
        if task == TaskType.SENTIMENT:
            return self._generate_sentiment_targets(processed_texts, task_config)
        elif task in [TaskType.NER, TaskType.POS]:
            return self._generate_sequence_labels(processed_texts, task_config, task.name.lower())
        elif task == TaskType.DISCOURSE:
            return self._generate_discourse_targets(processed_texts, task_config)
        elif task == TaskType.CONTRASTIVE:
            return self._generate_contrastive_targets(processed_texts, task_config)
        elif task in [TaskType.MLM, TaskType.LMLM]:
            return self._generate_mlm_lmlm_targets(processed_texts, task_config, task == TaskType.LMLM)
        elif task == TaskType.NSP:
            return self._generate_nsp_targets(processed_texts, task_config)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _generate_sentiment_targets(
        self,
        texts: List[str],
        task_config: Dict
    ) -> Tuple[np.ndarray, None, Optional[Dict[str, Any]]]:
        """Generate sentiment labels using pretrained model."""
        model, tokenizer = self.model_manager.get_model(
            "transformer", task_config["model"], task="sentiment"
        )
        
        encodings = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = outputs.logits.argmax(-1)
        
        return predictions.numpy(), None, None
    
    def _generate_sequence_labels(
        self,
        texts: List[str],
        task_config: Dict,
        task: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """Generate sequence labels for NER and POS tasks."""
        model_type = task_config.get("type", "transformer")
        model, tokenizer = self.model_manager.get_model(model_type, task_config["model"], task)
        
        if model_type == "spacy":
            return self._generate_spacy_labels(texts, model, task)
        return self._generate_transformer_labels(texts, model, tokenizer)
    
    def _generate_transformer_labels(
        self,
        texts: List[str],
        model: Any,
        tokenizer: Any
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """Generate labels using transformer models."""
        all_labels = []
        attention_masks = []
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1).squeeze(0)[1:-1].numpy()  # Remove CLS/SEP
            attention_mask = inputs["attention_mask"].squeeze(0)[1:-1].numpy()
            
            all_labels.append(self._pad_or_truncate(predictions.tolist()))
            attention_masks.append(self._pad_or_truncate(attention_mask.tolist()))
        
        return np.array(all_labels), np.array(attention_masks), None
    
    def _generate_spacy_labels(
        self,
        texts: List[str],
        nlp: Any,
        task: str
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """Generate labels using spaCy models."""
        all_labels = []
        masks = []
        
        for doc in nlp.pipe(texts):
            if task == "pos":
                labels = [token.pos for token in doc]
                mask = np.ones(len(doc))
            else:  # NER
                labels = []
                mask = np.zeros(len(doc))
                for ent in doc.ents:
                    labels.extend([doc.vocab.strings.add(ent.label_)] * (ent.end - ent.start))
                    mask[ent.start:ent.end] = 1
            
            all_labels.append(self._pad_or_truncate(labels))
            masks.append(self._pad_or_truncate(mask.tolist()))
        
        return np.array(all_labels), np.array(masks), None
    
    def _generate_mlm_lmlm_targets(
        self,
        texts: List[str],
        task_config: Dict,
        is_lmlm: bool = False,
        model_type: str = "transformer"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate masked language modeling targets using tokenization."""
        # Get tokenizer based on model type
        tokenizer_config = self.config['tokenizers'][model_type]
        tokenizer = self.model_manager.get_tokenizer(
            tokenizer_config['type'],
            tokenizer_config['model']
        )
        max_length = task_config.get('max_length', 512)
        
        # Create a tokenized copy for mask generation
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np',
            return_word_ids=True  # Added to get word_ids for alignment
        )
        
        input_shape = encoded['input_ids'].shape
        attention_mask = encoded['attention_mask']
        
        # Generate masks based on task type
        masks = self._generate_task_specific_masks(
            encoded,
            is_lmlm,
            task_config,
            model_type,
            attention_mask
        )
        
        # Enhanced metadata for alignment
        metadata = {
            'tokenization_info': {
                'model_type': model_type,
                'tokenizer_name': tokenizer_config['model'],
                'tokenizer_type': tokenizer_config['type'],
                'sequence_lengths': [len(t) for t in texts],
                'token_maps': encoded.word_ids() if hasattr(encoded, 'word_ids') else None,
                'original_text_hashes': [self._hash_text(t) for t in texts],
                'max_length': max_length,
                'attention_mask': attention_mask,
                'whole_word_mask': task_config.get('whole_word_mask', False) and model_type == "transformer",
                'special_tokens': {
                    'cls': tokenizer.cls_token_id,
                    'sep': tokenizer.sep_token_id,
                    'pad': tokenizer.pad_token_id,
                    'mask': tokenizer.mask_token_id
                }
            },
            'task_info': {
                'type': 'lmlm' if is_lmlm else 'mlm',
                'config': task_config
            }
        }
        
        return encoded['input_ids'], masks, metadata
    
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
    
    def reconstruct_text(self, target_ids: np.ndarray, metadata: Dict) -> List[str]:
        """Reconstruct original text from target IDs using metadata."""
        tokenizer_info = metadata['tokenization_info']
        tokenizer = self.model_manager.get_tokenizer(
            tokenizer_info['tokenizer_type'],
            tokenizer_info['tokenizer_name']
        )
        
        # Remove special tokens and padding
        special_tokens = tokenizer_info['special_tokens']
        special_token_ids = set(special_tokens.values())
        
        reconstructed_texts = []
        for seq_ids, mask in zip(target_ids, metadata['tokenization_info']['attention_mask']):
            # Filter out special tokens and padding
            valid_ids = [
                id_ for id_, is_valid in zip(seq_ids, mask)
                if id_ not in special_token_ids and is_valid
            ]
            
            # Decode tokens
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            reconstructed_texts.append(text)
        
        return reconstructed_texts
    
    def _generate_contrastive_targets(
        self,
        texts: List[str],
        task_config: Dict
    ) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Generate contrastive learning targets with validation metrics."""
        # Get embeddings
        nlp = spacy.load("en_core_web_lg")
        features = np.array([doc.vector for doc in nlp.pipe(texts)])
        
        min_clusters = task_config.get("min_clusters", 3)
        max_clusters = task_config.get("max_clusters", 8)
        min_score = task_config.get("min_silhouette_score", 0.5)
        
        best_score = -1
        best_labels = None
        best_metrics = {}
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            
            if silhouette > best_score:
                best_score = silhouette
                best_labels = labels
                best_metrics = {
                    "silhouette_score": silhouette,
                    "calinski_harabasz_score": calinski,
                    "n_clusters": n_clusters,
                    "cluster_centers": kmeans.cluster_centers_
                }
            
            if best_score >= min_score:
                break
        
        return best_labels, None, best_metrics

class TargetProcessor:
    def __init__(
        self,
        config_manager,
        task_name: str
    ):
        self.config = config_manager.get_config("data_config")
        self.task_config = self.config["label_generation"]["models"][task_name]
    
    def process_targets(
        self,
        targets: Union[np.ndarray, List[Any]],
        task: str
    ) -> Dict[str, tf.Tensor]:
        """Process raw targets into model-ready format."""
        if task in ["sentiment", "contrastive"]:
            return self._process_classification_targets(targets)
        elif task in ["ner", "pos"]:
            return self._process_token_targets(targets, task)
        elif task == "discourse":
            return self._process_discourse_targets(targets)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _process_classification_targets(
        self,
        targets: np.ndarray
    ) -> Dict[str, tf.Tensor]:
        """Process classification targets."""
        return {
            "labels": tf.convert_to_tensor(targets, dtype=tf.int32)
        }
    
    def _process_token_targets(
        self,
        targets: List[List[str]],
        task: str
    ) -> Dict[str, tf.Tensor]:
        """Process token-level targets."""
        # Create vocabulary for labels
        label_set = set()
        for seq in targets:
            label_set.update(seq)
        label_to_id = {label: i for i, label in enumerate(sorted(label_set))}
        
        # Convert to indices
        target_ids = [
            [label_to_id[label] for label in seq]
            for seq in targets
        ]
        
        return {
            "labels": tf.ragged.constant(target_ids).to_tensor(),
            "label_to_id": tf.constant(list(label_to_id.keys()))
        }
    
    def _process_discourse_targets(
        self,
        targets: List[List[str]]
    ) -> Dict[str, tf.Tensor]:
        """Process discourse markers."""
        markers = self.task_config["markers"]
        marker_to_id = {marker: i for i, marker in enumerate(markers)}
        
        # Create multi-hot encoding
        multi_hot = np.zeros((len(targets), len(markers)))
        for i, seq in enumerate(targets):
            for marker in seq:
                if marker in marker_to_id:
                    multi_hot[i, marker_to_id[marker]] = 1
        
        return {
            "labels": tf.convert_to_tensor(multi_hot, dtype=tf.float32),
            "marker_to_id": tf.constant(markers)
        }

class TargetHandler:
    """Handles generation and processing of target labels."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        model_manager: ModelManager
    ):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_config = config_manager.get_config('data_config')
    
    def generate_targets(
        self,
        texts: List[str],
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> ModelTarget:
        """Generate targets for specified tasks."""
        dataset_config = self.data_config['datasets'][dataset_name]
        max_length = dataset_config.get('max_length', 512)
        
        # Initialize ModelTarget with reconstruction targets
        tokenizer = self._get_tokenizer(model_type, dataset_config)
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        model_target = ModelTarget(
            reconstruction_targets=encodings['input_ids'].numpy(),
            sequence_mask=encodings['attention_mask'].numpy(),
            metadata={
                'tokenizer_info': {
                    'type': model_type,
                    'name': dataset_config['tokenizer']['model'],
                    'special_tokens': get_special_token_info(tokenizer)
                },
                'text_hashes': [hash_text(text) for text in texts],
                'sequence_lengths': [len(text.split()) for text in texts],
                'processing_time': datetime.now().isoformat()
            }
        )
        
        # Get enabled tasks if none specified
        if tasks is None:
            tasks = [TaskType[task.upper()] for task in dataset_config['enabled_tasks']]
        
        # Generate targets for each task
        for task in tasks:
            task_config = self._get_task_config(task, dataset_config)
            if task_config:
                labels, mask, metadata = self._generate_task_targets(
                    texts,
                    task,
                    task_config,
                    model_type
                )
                model_target.add_task_labels(task.name.lower(), labels, mask, metadata)
        
        return model_target
    
    def _get_tokenizer(self, model_type: str, dataset_config: Dict):
        """Get appropriate tokenizer based on model type and config."""
        tokenizer_config = dataset_config.get('tokenizer', {})
        return self.model_manager.get_tokenizer(
            tokenizer_type=tokenizer_config.get('type', 'wordpiece'),
            model_name=tokenizer_config.get('model', 'bert-base-uncased')
        )
    
    def _get_task_config(self, task: TaskType, dataset_config: Dict) -> Optional[Dict]:
        """Get task-specific configuration."""
        task_name = task.name.lower()
        if task_name not in dataset_config.get('enabled_tasks', []):
            return None
        
        task_config = self.data_config['tasks'].get(task_name, {})
        task_overrides = dataset_config.get('task_overrides', {}).get(task_name, {})
        
        # Merge base config with overrides
        return {**task_config, **task_overrides}
    
    def _generate_task_targets(
        self,
        texts: List[str],
        task: TaskType,
        config: Dict,
        model_type: str = "transformer"
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Generate targets for a specific task."""
        if task == TaskType.SENTIMENT:
            return self._generate_sentiment_targets(texts, config)
        elif task in [TaskType.NER, TaskType.POS]:
            return self._generate_token_targets(texts, task, config)
        elif task == TaskType.DISCOURSE:
            return self._generate_discourse_targets(texts, config)
        elif task == TaskType.CONTRASTIVE:
            return self._generate_contrastive_targets(texts, config)
        elif task in [TaskType.MLM, TaskType.LMLM]:
            return self._generate_mlm_lmlm_targets(
                texts,
                is_lmlm=(task == TaskType.LMLM),
                task_config=config,
                model_type=model_type
            )
        elif task == TaskType.NSP:
            return self._generate_nsp_targets(texts, config)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _generate_sentiment_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Optional[Dict[str, Any]]]:
        """Generate sentiment analysis targets."""
        model = self.model_manager.get_model(
            model_type="transformer",
            model_name=config['model'],
            task="sentiment"
        )
        
        outputs = model(texts)
        labels = np.array([output['label'] for output in outputs])
        
        metadata = {
            'model_name': config['model'],
            'label_mapping': config.get('label_mapping', {}),
            'preserve_original_labels': config.get('preserve_original_labels', True)
        }
        
        return labels, None, metadata
    
    def _generate_token_targets(
        self,
        texts: List[str],
        task: TaskType,
        config: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """Generate token-level targets (NER, POS)."""
        model_type = config.get('type', 'spacy')
        model = self.model_manager.get_model(
            model_type=model_type,
            model_name=config['model'],
            task=task.name.lower()
        )
        
        if model_type == "spacy":
            docs = list(model.pipe(texts))
            labels = []
            masks = []
            
            for doc in docs:
                if task == TaskType.POS:
                    doc_labels = [token.pos_ for token in doc]
                    doc_mask = np.ones(len(doc))
                else:  # NER
                    doc_labels = ["O"] * len(doc)
                    doc_mask = np.zeros(len(doc))
                    for ent in doc.ents:
                        doc_labels[ent.start:ent.end] = [f"B-{ent.label_}"] + [f"I-{ent.label_}"] * (ent.end - ent.start - 1)
                        doc_mask[ent.start:ent.end] = 1
                
                labels.append(doc_labels)
                masks.append(doc_mask)
            
            metadata = {
                'model_name': config['model'],
                'align_with_tokens': config.get('align_with_tokens', True),
                'label_scheme': 'BIO' if task == TaskType.NER else 'POS'
            }
            
            return np.array(labels), np.array(masks), metadata
        
        else:
            raise ValueError(f"Unsupported model type for {task.name}: {model_type}")
    
    def _generate_discourse_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Generate discourse marker targets."""
        markers = config.get('markers', [])
        labels = np.zeros((len(texts), len(markers)))
        
        for i, text in enumerate(texts):
            for j, marker in enumerate(markers):
                if marker.lower() in text.lower():
                    labels[i, j] = 1
        
        metadata = {
            'markers': markers,
            'multi_label': True,
            'label_distribution': {
                marker: int(labels[:, i].sum())
                for i, marker in enumerate(markers)
            }
        }
        
        return labels, None, metadata
    
    def _generate_contrastive_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Generate contrastive learning targets."""
        # Get embeddings for clustering
        model = self.model_manager.get_model(
            model_type="transformer",
            model_name=config['model']
        )
        
        embeddings = model.encode(texts)
        
        # Find optimal number of clusters
        min_clusters = config.get('min_clusters', 3)
        max_clusters = config.get('max_clusters', 8)
        n_clusters = self._find_optimal_clusters(
            embeddings,
            min_clusters,
            max_clusters
        )
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        metadata = {
            'n_clusters': n_clusters,
            'cluster_sizes': np.bincount(cluster_labels).tolist(),
            'silhouette_score': float(silhouette_score(embeddings, cluster_labels)),
            'clustering_method': config.get('clustering_method', 'kmeans')
        }
        
        return cluster_labels, None, metadata
    
    def _find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int
    ) -> int:
        """Find optimal number of clusters using silhouette score."""
        scores = []
        for n in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append((score, n))
        
        return max(scores, key=lambda x: x[0])[1]
    
    def _generate_mlm_lmlm_targets(
        self,
        texts: List[str],
        is_lmlm: bool = False,
        task_config: Dict = None,
        model_type: str = "transformer"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate masked language modeling targets."""
        tokenizer = self._get_tokenizer(model_type, task_config)
        max_length = task_config.get('max_length', 512)
        
        # Tokenize texts
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        input_ids = encodings['input_ids'].numpy()
        attention_mask = encodings['attention_mask'].numpy()
        
        # Generate masks
        if is_lmlm:
            masked_positions = self._generate_lmlm_masks(
                input_ids,
                attention_mask,
                task_config
            )
        else:
            masked_positions = self._generate_mlm_masks(
                input_ids,
                attention_mask,
                task_config
            )
        
        metadata = {
            'task_type': 'lmlm' if is_lmlm else 'mlm',
            'mask_probability': task_config.get('mask_probability', 0.15),
            'whole_word_mask': task_config.get('whole_word_mask', True),
            'masked_positions': masked_positions.tolist()
        }
        
        return input_ids, masked_positions, metadata
    
    def _generate_mlm_masks(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Generate masks for standard MLM."""
        probability = config.get('mask_probability', 0.15)
        special_tokens = config.get('special_tokens', [])
        
        masked_positions = np.zeros_like(input_ids)
        valid_positions = (attention_mask == 1) & ~np.isin(input_ids, special_tokens)
        
        for i in range(len(input_ids)):
            valid_indices = np.where(valid_positions[i])[0]
            n_mask = int(len(valid_indices) * probability)
            mask_indices = np.random.choice(valid_indices, n_mask, replace=False)
            masked_positions[i, mask_indices] = 1
        
        return masked_positions
    
    def _generate_lmlm_masks(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Generate masks for large-span MLM."""
        min_span = config.get('min_span', 2)
        max_span = config.get('max_span', 5)
        special_tokens = config.get('special_tokens', [])
        
        masked_positions = np.zeros_like(input_ids)
        valid_positions = (attention_mask == 1) & ~np.isin(input_ids, special_tokens)
        
        for i in range(len(input_ids)):
            valid_indices = np.where(valid_positions[i])[0]
            current_pos = 0
            
            while current_pos < len(valid_indices):
                if np.random.random() < config.get('mask_probability', 0.15):
                    span_length = np.random.randint(min_span, max_span + 1)
                    end_pos = min(current_pos + span_length, len(valid_indices))
                    masked_positions[i, valid_indices[current_pos:end_pos]] = 1
                    current_pos = end_pos
                else:
                    current_pos += 1
        
        return masked_positions
    
    def _generate_nsp_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, None]:
        """Generate next sentence prediction targets."""
        # Split texts into pairs
        text_pairs = []
        labels = []
        
        for i in range(0, len(texts) - 1, 2):
            if np.random.random() < config.get('negative_sampling_ratio', 0.5):
                # Create negative pair
                random_idx = np.random.choice(
                    [j for j in range(len(texts)) if j != i + 1]
                )
                text_pairs.append((texts[i], texts[random_idx]))
                labels.append(0)
            else:
                # Use consecutive sentences
                text_pairs.append((texts[i], texts[i + 1]))
                labels.append(1)
        
        return np.array(labels), None, None 