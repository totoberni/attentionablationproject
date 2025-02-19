from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
from transformers import AutoTokenizer
import tensorflow_datasets as tfds
from pathlib import Path

class DatasetSetup:
    def __init__(
        self,
        config_manager,
        dataset_name: str,
        split: str = "train"
    ):
        self.config = config_manager.get_config("data_config")
        self.dataset_config = self.config["datasets"][dataset_name]
        self.split = split
        
        # Initialize preprocessing components
        self.text_cleaner = self._setup_text_cleaner()
    
    def _setup_text_cleaner(self):
        """Setup text cleaner based on configuration."""
        preprocessing_config = self.dataset_config["preprocessing"]
        
        def clean_text(text: tf.Tensor) -> tf.Tensor:
            # Only perform text cleaning, no tokenization
            if preprocessing_config["remove_html"]:
                text = tf.strings.regex_replace(text, '<[^>]*>', ' ')
            if preprocessing_config["normalize_unicode"]:
                text = tf.strings.unicode_normalize('NFKC', text)
            if preprocessing_config["handle_numbers"]:
                text = tf.strings.regex_replace(text, r'\d+', '[NUM]')
            text = tf.strings.regex_replace(text, r'\s+', ' ')
            return tf.strings.strip(text)
        
        return clean_text
    
    def prepare_dataset(
        self,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True
    ) -> tf.data.Dataset:
        """Prepare dataset for training/evaluation."""
        # Load raw dataset
        dataset = self._load_dataset()
        
        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if cache:
            dataset = dataset.cache()
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_dataset(self) -> tf.data.Dataset:
        """Load dataset from source."""
        try:
            dataset = tfds.load(
                self.dataset_config["name"],
                split=self.split,
                as_supervised=True
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}")
        
        return dataset
    
    def _preprocess_example(
        self,
        text: tf.Tensor,
        label: Optional[tf.Tensor] = None
    ) -> Dict[str, tf.Tensor]:
        """Preprocess a single example."""
        # Only clean text, no tokenization
        text = self.text_cleaner(text)
        
        features = {
            "text": text
        }
        
        if label is not None:
            features["labels"] = label
        
        return features

    def preprocess_text(self, text: tf.Tensor) -> tf.Tensor:
        """Preprocess text according to configuration."""
        # Remove HTML if configured
        if self.config.get('remove_html', False):
            text = tf.strings.regex_replace(text, '<[^>]+>', '')
        
        # Normalize unicode if configured
        if self.config.get('normalize_unicode', False):
            text = tf.strings.unicode_normalize('NFKC', text)
        
        # Handle numbers if configured
        if self.config.get('handle_numbers', False):
            text = tf.strings.regex_replace(text, r'\d+', '[NUM]')
        
        return text 