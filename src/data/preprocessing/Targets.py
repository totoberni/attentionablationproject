from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
from transformers import pipeline
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from src.data.core import ConfigurationManager, ModelManager

class TargetGenerator:
    def __init__(
        self,
        config_manager,
        task_name: str
    ):
        self.config = config_manager.get_config("data_config")
        self.task_config = self.config["label_generation"]["models"][task_name]
        self.model = self._setup_model()
    
    def _setup_model(self) -> Any:
        """Setup the appropriate model for target generation."""
        if self.task_config["type"] == "transformer":
            return pipeline(
                task=self.task_config.get("task", "text-classification"),
                model=self.task_config["model"],
                device=0 if tf.config.list_physical_devices('GPU') else -1
            )
        elif self.task_config["type"] == "spacy":
            return spacy.load(self.task_config["model"])
        elif self.task_config["type"] == "sklearn":
            return self._setup_clustering_model()
        else:
            raise ValueError(f"Unsupported model type: {self.task_config['type']}")
    
    def _setup_clustering_model(self) -> KMeans:
        """Setup clustering model for contrastive learning."""
        return KMeans(
            n_clusters=self.task_config["min_clusters"],
            random_state=42
        )
    
    def generate_targets(
        self,
        texts: List[str],
        task: str
    ) -> Union[np.ndarray, List[Any]]:
        """Generate target labels for the given texts."""
        if task == "sentiment":
            return self._generate_sentiment_targets(texts)
        elif task in ["ner", "pos"]:
            return self._generate_token_targets(texts, task)
        elif task == "discourse":
            return self._generate_discourse_targets(texts)
        elif task == "contrastive":
            return self._generate_contrastive_targets(texts)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _generate_sentiment_targets(self, texts: List[str]) -> np.ndarray:
        """Generate sentiment labels using transformer model."""
        results = self.model(texts)
        return np.array([r["label"] for r in results])
    
    def _generate_token_targets(
        self,
        texts: List[str],
        task: str
    ) -> List[List[str]]:
        """Generate token-level targets (NER or POS tags)."""
        targets = []
        for doc in self.model.pipe(texts):
            if task == "ner":
                tokens = [ent.label_ for ent in doc.ents]
            else:  # POS
                tokens = [token.pos_ for token in doc]
            targets.append(tokens)
        return targets
    
    def _generate_discourse_targets(self, texts: List[str]) -> List[str]:
        """Generate discourse markers."""
        markers = self.task_config["markers"]
        targets = []
        
        for text in texts:
            found_markers = []
            for marker in markers:
                if marker.lower() in text.lower():
                    found_markers.append(marker)
            targets.append(found_markers)
        
        return targets
    
    def _generate_contrastive_targets(self, texts: List[str]) -> np.ndarray:
        """Generate contrastive learning targets using clustering."""
        # Get embeddings using spaCy
        nlp = spacy.load("en_core_web_lg")
        embeddings = np.array([doc.vector for doc in nlp.pipe(texts)])
        
        # Find optimal number of clusters
        best_n_clusters = self._find_optimal_clusters(
            embeddings,
            self.task_config["min_clusters"],
            self.task_config["max_clusters"]
        )
        
        # Fit clustering model
        self.model.n_clusters = best_n_clusters
        return self.model.fit_predict(embeddings)
    
    def _find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int
    ) -> int:
        """Find optimal number of clusters using silhouette score."""
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
    
    def generate_targets(self, texts: List[str], dataset_name: str) -> Dict[str, np.ndarray]:
        """Generate targets based on dataset configuration."""
        dataset_config = self.data_config['datasets'][dataset_name]
        
        targets = {}
        for task_name, task_config in dataset_config.get('tasks', {}).items():
            if task_config.get('enabled', False):
                task_targets = self._generate_task_targets(texts, task_name, task_config)
                targets[task_name] = task_targets
        
        return targets
    
    def _generate_task_targets(self, texts: List[str], task_name: str, 
                             task_config: Dict) -> np.ndarray:
        """Generate targets for a specific task."""
        model_type = task_config['type']
        model_name = task_config['model']
        
        model = self.model_manager.get_model(model_type, model_name, task_name)[0]
        # Implementation of task-specific target generation
        return np.array([])  # Placeholder for actual implementation 