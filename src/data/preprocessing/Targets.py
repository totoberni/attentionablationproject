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

from src.data.core import (
    ConfigurationManager,
    ModelManager,
    TaskType,
    ModelTarget,
    TaskLabels
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
            return_tensors='np'
        )
        
        input_shape = encoded['input_ids'].shape
        attention_mask = encoded['attention_mask']
        
        if is_lmlm:
            min_span = task_config.get('min_span', 2)
            max_span = task_config.get('max_span', 5)
            min_masks = task_config.get('min_masks', 1)
            max_masks = task_config.get('max_masks', 5)
            
            # Generate span masks respecting attention mask
            masks = np.zeros(input_shape, dtype=bool)
            for i in range(input_shape[0]):
                valid_length = attention_mask[i].sum()
                num_masks = np.random.randint(min_masks, max_masks + 1)
                
                for _ in range(num_masks):
                    span_length = np.random.randint(min_span, min(max_span + 1, valid_length))
                    start = np.random.randint(0, valid_length - span_length)
                    masks[i, start:start+span_length] = True
        else:
            # Standard MLM with attention mask
            mask_prob = task_config['mask_probability']
            masks = np.random.random(input_shape) < mask_prob
            masks = masks & (attention_mask == 1)  # Only mask actual tokens
            
            if task_config.get('whole_word_mask', False) and model_type == "transformer":
                # Use tokenizer's word IDs for whole word masking (transformer only)
                word_ids = [tokenizer.get_word_ids(ids) for ids in encoded['input_ids']]
                for i, word_id_seq in enumerate(word_ids):
                    masked_words = set()
                    for j, word_id in enumerate(word_id_seq):
                        if word_id is not None:
                            if word_id in masked_words:
                                masks[i, j] = True
                            elif masks[i, j]:
                                masked_words.add(word_id)
        
        # Store tokenization info for potential alignment checks
        metadata = {
            'mask_type': 'lmlm' if is_lmlm else 'mlm',
            'model_type': model_type,
            'tokenizer_name': tokenizer_config['model'],
            'tokenizer_type': tokenizer_config['type'],
            'max_length': max_length,
            'attention_mask': attention_mask,
            'whole_word_mask': task_config.get('whole_word_mask', False) and model_type == "transformer"
        }
        
        return encoded['input_ids'], masks, metadata
    
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
    
    def __init__(self, config_manager: ConfigurationManager, model_manager: ModelManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_config = config_manager.get_config('data')
    
    def generate_targets(
        self,
        texts: List[str],
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> ModelTarget:
        """Generate targets based on dataset configuration."""
        # Create a copy of texts for processing to avoid modifying originals
        processed_texts = texts.copy()
        dataset_config = self.data_config['datasets'][dataset_name]
        
        # Initialize ModelTarget
        model_target = ModelTarget(
            reconstruction_targets=np.zeros((len(processed_texts), dataset_config['max_length'])),
            sequence_mask=np.ones((len(processed_texts), dataset_config['max_length']))
        )
        
        # If no tasks specified, use enabled tasks from config
        if tasks is None:
            tasks = [TaskType[task.upper()] for task in dataset_config['enabled_tasks']]
        
        # Generate targets for each task
        for task in tasks:
            task_config = dataset_config['task_overrides'].get(
                task.name.lower(),
                self.data_config['tasks'][task.name.lower()]
            )
            
            if task_config.get('enabled', True):
                labels, mask, metadata = self._generate_task_targets(
                    processed_texts.copy(),  # Pass a new copy for each task
                    task,
                    task_config,
                    model_type
                )
                
                model_target.add_task_labels(
                    task_type=task,
                    labels=labels,
                    mask=mask,
                    metadata=metadata
                )
        
        return model_target
    
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
            return self._generate_mlm_lmlm_targets(texts, task == TaskType.LMLM, config, model_type)
        elif task == TaskType.NSP:
            return self._generate_nsp_targets(texts, config)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _generate_sentiment_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Optional[Dict[str, Any]]]:
        model = self.model_manager.get_model(config['type'], config['model'])
        results = model(texts)
        return np.array([r['label'] for r in results]), None, None
    
    def _generate_token_targets(
        self,
        texts: List[str],
        task: TaskType,
        config: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        model = self.model_manager.get_model(config['type'], config['model'])
        
        # Process texts and get token-level predictions
        targets = []
        masks = []
        
        for doc in model.pipe(texts):
            if task == TaskType.NER:
                tokens = [ent.label_ for ent in doc.ents]
                mask = np.zeros(len(doc))
                for ent in doc.ents:
                    mask[ent.start:ent.end] = 1
            else:  # POS
                tokens = [token.pos_ for token in doc]
                mask = np.ones(len(doc))
            
            targets.append(tokens)
            masks.append(mask)
        
        return np.array(targets), np.array(masks), None
    
    def _generate_discourse_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        markers = config['markers']
        targets = np.zeros((len(texts), len(markers)))
        
        for i, text in enumerate(texts):
            for j, marker in enumerate(markers):
                if marker in text:
                    targets[i, j] = 1
        
        return targets, None, {'markers': markers}
    
    def _generate_contrastive_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        # Get embeddings
        nlp = spacy.load('en_core_web_lg')
        embeddings = np.array([doc.vector for doc in nlp.pipe(texts)])
        
        # Find optimal clusters
        n_clusters = self._find_optimal_clusters(
            embeddings,
            config['min_clusters'],
            config['max_clusters']
        )
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        return labels, None, {
            'embeddings': embeddings,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def _generate_mlm_lmlm_targets(
        self,
        texts: List[str],
        is_lmlm: bool = False,
        task_config: Dict = None,
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
            return_tensors='np'
        )
        
        input_shape = encoded['input_ids'].shape
        attention_mask = encoded['attention_mask']
        
        if is_lmlm:
            min_span = task_config.get('min_span', 2)
            max_span = task_config.get('max_span', 5)
            min_masks = task_config.get('min_masks', 1)
            max_masks = task_config.get('max_masks', 5)
            
            # Generate span masks respecting attention mask
            masks = np.zeros(input_shape, dtype=bool)
            for i in range(input_shape[0]):
                valid_length = attention_mask[i].sum()
                num_masks = np.random.randint(min_masks, max_masks + 1)
                
                for _ in range(num_masks):
                    span_length = np.random.randint(min_span, min(max_span + 1, valid_length))
                    start = np.random.randint(0, valid_length - span_length)
                    masks[i, start:start+span_length] = True
        else:
            # Standard MLM with attention mask
            mask_prob = task_config['mask_probability']
            masks = np.random.random(input_shape) < mask_prob
            masks = masks & (attention_mask == 1)  # Only mask actual tokens
            
            if task_config.get('whole_word_mask', False) and model_type == "transformer":
                # Use tokenizer's word IDs for whole word masking (transformer only)
                word_ids = [tokenizer.get_word_ids(ids) for ids in encoded['input_ids']]
                for i, word_id_seq in enumerate(word_ids):
                    masked_words = set()
                    for j, word_id in enumerate(word_id_seq):
                        if word_id is not None:
                            if word_id in masked_words:
                                masks[i, j] = True
                            elif masks[i, j]:
                                masked_words.add(word_id)
        
        # Store tokenization info for potential alignment checks
        metadata = {
            'mask_type': 'lmlm' if is_lmlm else 'mlm',
            'model_type': model_type,
            'tokenizer_name': tokenizer_config['model'],
            'tokenizer_type': tokenizer_config['type'],
            'max_length': max_length,
            'attention_mask': attention_mask,
            'whole_word_mask': task_config.get('whole_word_mask', False) and model_type == "transformer"
        }
        
        return encoded['input_ids'], masks, metadata
    
    def _generate_nsp_targets(
        self,
        texts: List[str],
        config: Dict
    ) -> Tuple[np.ndarray, None, None]:
        # Split into pairs
        pairs = [(texts[i], texts[i+1]) for i in range(0, len(texts)-1, 2)]
        labels = []
        
        for first, second in pairs:
            if np.random.random() < config['negative_sampling_ratio']:
                # Create negative pair
                random_idx = np.random.randint(len(texts))
                labels.append(0)
            else:
                labels.append(1)
        
        return np.array(labels), None, None
    
    def _find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int
    ) -> int:
        best_score = -1
        best_n = min_clusters
        
        for n in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_n = n
        
        return best_n 