import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class BaseManager:
    """Base class for all managers providing common functionality."""
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup basic logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

class ConfigurationManager(BaseManager):
    """Manages loading and validation of configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        super().__init__("ConfigManager")
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all YAML configuration files from the config directory."""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration: {config_name}")
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get a specific configuration by name."""
        if name not in self.configs:
            raise ValueError(f"Configuration '{name}' not found")
        return self.configs[name]
    
    def validate_config(self, name: str, schema: Dict[str, Any]) -> bool:
        """Validate a configuration against a schema."""
        if name not in self.configs:
            raise ValueError(f"Configuration '{name}' not found")
        
        config = self.configs[name]
        try:
            self._validate_dict(config, schema)
            return True
        except ValueError as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _validate_dict(self, config: Dict[str, Any], schema: Dict[str, Any], path: str = ""):
        """Recursively validate configuration dictionary against schema."""
        for key, value_type in schema.items():
            if key not in config:
                raise ValueError(f"Missing required key '{path}{key}'")
            
            if isinstance(value_type, dict):
                if not isinstance(config[key], dict):
                    raise ValueError(f"Expected dict for '{path}{key}'")
                self._validate_dict(config[key], value_type, f"{path}{key}.")
            elif isinstance(value_type, type):
                if not isinstance(config[key], value_type):
                    raise ValueError(
                        f"Invalid type for '{path}{key}': "
                        f"expected {value_type.__name__}, got {type(config[key]).__name__}"
                    )
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """Update a configuration with new values."""
        if name not in self.configs:
            raise ValueError(f"Configuration '{name}' not found")
        
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        
        deep_update(self.configs[name], updates)
        
        # Save updated config
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.configs[name], f, default_flow_style=False)
        self.logger.info(f"Updated and saved configuration: {name}")

class CacheManager(BaseManager):
    """Manages caching of models, tokenizers, and other resources."""
    
    def __init__(self, cache_dir: str, max_size: int = 5):
        super().__init__("CacheManager")
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        """Get the cache path for a given key."""
        return self.cache_dir / key
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return (self.cache_dir / key).exists()
    
    def clear(self, key: Optional[str] = None):
        """Clear specific or all cached items."""
        if key:
            cache_path = self.cache_dir / key
            if cache_path.exists():
                if cache_path.is_file():
                    cache_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(cache_path)
                self.logger.info(f"Cleared cache for: {key}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
            self.logger.info("Cleared all cache") 