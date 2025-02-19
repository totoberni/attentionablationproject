import yaml
from typing import Dict, Any, Optional
import os
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all YAML configuration files from the config directory."""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = yaml.safe_load(f)
    
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
            print(f"Configuration validation failed: {str(e)}")
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