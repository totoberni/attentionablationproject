from typing import Dict, Type, Optional, Any
import tensorflow as tf
from pathlib import Path
import os
import logging
from functools import lru_cache

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

class ModelManager:
    """Manager for loading, saving, and caching model instances."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        cache_dir: str = "cache/models",
        max_cache_size: int = 5
    ):
        self.registry = registry
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Configure model caching
        self.load_model = lru_cache(maxsize=max_cache_size)(self._load_model)
    
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
    
    def _load_model(
        self,
        name: str,
        weights_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> tf.keras.Model:
        """Load a model with caching."""
        model = self.create_model(name, config or {})
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
    
    def clear_cache(self):
        """Clear the model cache."""
        self.load_model.cache_clear()
    
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
        return cached 