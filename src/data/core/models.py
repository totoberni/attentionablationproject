from typing import Dict, Type, Optional, Any, Tuple, Union
import tensorflow as tf
from pathlib import Path
import os
import logging
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel, TFAutoModel, pipeline
import spacy
import sentencepiece as spm
from .base import BaseManager, ConfigurationManager, CacheManager

class ModelRegistry:
    """Registry for model architectures and their configurations."""
    
    def __init__(self):
        self._models = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, name: str, model_class: Type[tf.keras.Model]):
        """Register a model class with a name."""
        if name in self._models:
            self.logger.warning(f"Overwriting existing model: {name}")
        self._models[name] = model_class
    
    def get_model_class(self, name: str) -> Type[tf.keras.Model]:
        """Get a model class by name."""
        if name not in self._models:
            raise ValueError(f"Model not found: {name}")
        return self._models[name]
    
    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def unregister(self, name: str):
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]

class ModelManager(BaseManager):
    """Manager for loading, saving, and caching model instances."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        registry: Optional[ModelRegistry] = None
    ):
        super().__init__("ModelManager")
        self.config_manager = config_manager
        self.registry = registry or ModelRegistry()
        
        # Get configuration
        model_config = config_manager.get_config('model_config')
        self.cache_manager = CacheManager(
            cache_dir=model_config.get('cache_dir', 'cache/models'),
            max_size=model_config.get('max_cache_size', 5)
        )
    
    @lru_cache(maxsize=5)
    def get_tokenizer(
        self,
        tokenizer_type: str,
        model_name: str,
        use_cache: bool = True
    ) -> Any:
        """Get tokenizer instance with caching."""
        cache_key = f"tokenizer_{tokenizer_type}_{model_name}"
        
        if use_cache and self.cache_manager.exists(cache_key):
            self.logger.info(f"Loading tokenizer from cache: {cache_key}")
            return self._load_from_cache(cache_key)
        
        tokenizer = self._create_tokenizer(tokenizer_type, model_name)
        
        if use_cache:
            self._save_to_cache(tokenizer, cache_key)
        
        return tokenizer
    
    def _create_tokenizer(self, tokenizer_type: str, model_name: str) -> Any:
        """Create a new tokenizer instance."""
        self.logger.info(f"Creating new tokenizer: {tokenizer_type} - {model_name}")
        
        if tokenizer_type == "wordpiece":
            return AutoTokenizer.from_pretrained(model_name)
        elif tokenizer_type == "sentencepiece":
            sp = spm.SentencePieceProcessor()
            sp.Load(f"{model_name}.model")
            return sp
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    def get_model(
        self,
        model_type: str,
        model_name: str,
        task: Optional[str] = None,
        use_cache: bool = True
    ) -> Union[tf.keras.Model, Tuple[tf.keras.Model, Any]]:
        """Get model instance with optional tokenizer."""
        cache_key = f"model_{model_type}_{model_name}_{task or 'base'}"
        
        if use_cache and self.cache_manager.exists(cache_key):
            self.logger.info(f"Loading model from cache: {cache_key}")
            return self._load_from_cache(cache_key)
        
        if model_type == "transformer":
            model = self._load_transformer_model(model_name, task)
        elif model_type == "spacy":
            model = self._load_spacy_model(model_name)
        else:
            model_class = self.registry.get_model_class(model_type)
            model = model_class()
        
        if use_cache:
            self._save_to_cache(model, cache_key)
        
        return model
    
    def _load_transformer_model(
        self,
        model_name: str,
        task: Optional[str] = None
    ) -> Union[tf.keras.Model, Tuple[tf.keras.Model, Any]]:
        """Load a Hugging Face transformer model."""
        self.logger.info(f"Loading transformer model: {model_name} for task: {task}")
        
        if task == "sentiment":
            return pipeline("sentiment-analysis", model=model_name)
        elif task == "ner":
            return pipeline("ner", model=model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModel.from_pretrained(model_name)
            return model, tokenizer
    
    def _load_spacy_model(self, model_name: str) -> Any:
        """Load a spaCy model."""
        self.logger.info(f"Loading spaCy model: {model_name}")
        return spacy.load(model_name)
    
    def _save_to_cache(self, obj: Any, key: str):
        """Save object to cache."""
        cache_path = self.cache_manager.get_cache_path(key)
        if hasattr(obj, 'save_pretrained'):
            obj.save_pretrained(str(cache_path))
        else:
            import joblib
            joblib.dump(obj, cache_path)
        self.logger.info(f"Saved to cache: {key}")
    
    def _load_from_cache(self, key: str) -> Any:
        """Load object from cache."""
        cache_path = self.cache_manager.get_cache_path(key)
        if not cache_path.exists():
            raise ValueError(f"Cache not found: {key}")
        
        if cache_path.is_dir():
            # Assume it's a Hugging Face model/tokenizer
            if "tokenizer" in key:
                return AutoTokenizer.from_pretrained(str(cache_path))
            else:
                return TFAutoModel.from_pretrained(str(cache_path))
        else:
            # Assume it's a joblib dump
            import joblib
            return joblib.load(cache_path)
    
    def clear_cache(self, key: Optional[str] = None):
        """Clear model cache."""
        self.cache_manager.clear(key)
        if key:
            self.get_tokenizer.cache_clear()  # Clear LRU cache
    
    def create_model(
        self,
        name: str,
        config: Dict[str, Any],
        weights_path: Optional[str] = None
    ) -> tf.keras.Model:
        """Create a new model instance."""
        model_class = self.registry.get_model_class(name)
        model = model_class(**config)
        
        if weights_path:
            self._load_weights(model, weights_path)
        
        return model
    
    def _load_weights(self, model: tf.keras.Model, weights_path: str):
        """Load weights into a model."""
        try:
            model.load_weights(weights_path)
        except Exception as e:
            self.logger.error(f"Failed to load weights from {weights_path}: {str(e)}")
            raise
    
    def save_model(
        self,
        model: tf.keras.Model,
        name: str,
        version: str = "latest"
    ):
        """Save a model's weights and config."""
        save_dir = self.cache_dir / name / version
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        weights_path = save_dir / "weights.h5"
        model.save_weights(str(weights_path))
        
        # Save config
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            f.write(model.to_json())
        
        self.logger.info(f"Saved model {name} (version: {version})")
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version number for a model."""
        model_dir = self.cache_dir / name
        if not model_dir.exists():
            return None
        
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if not versions:
            return None
        
        try:
            # Try to sort as version numbers
            versions.sort(key=lambda x: [int(p) for p in x.split('.')])
        except ValueError:
            # Fall back to string sorting
            versions.sort()
        
        return versions[-1]
    
    def list_cached_models(self) -> Dict[str, list[str]]:
        """List all cached models and their versions."""
        cached = {}
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                versions = [
                    d.name for d in model_dir.iterdir()
                    if d.is_dir() and (d / "weights.h5").exists()
                ]
                if versions:
                    cached[model_dir.name] = versions 