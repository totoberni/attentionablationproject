from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import yaml
import argparse
import logging
from google.cloud import storage
import json
import math
from .base import BaseManager, ConfigurationManager
from .models import ModelManager

logger = logging.getLogger(__name__)

class DatasetSetup(BaseManager):
    """Handles dataset setup and preprocessing."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        model_manager: ModelManager,
        dataset_name: str
    ):
        super().__init__("DatasetSetup")
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.dataset_name = dataset_name
        
        # Load dataset configuration
        self.dataset_config = self.config_manager.get_config(f'datasets/{dataset_name}')
        if not self.dataset_config:
            raise ValueError(f"Dataset configuration not found: {dataset_name}")
        
        # Set up paths
        self.output_dir = Path(self.dataset_config.get('output_dir', 'data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure sharding
        self.num_shards = self.dataset_config.get('num_shards', 10)
        self.shuffle_buffer_size = self.dataset_config.get('shuffle_buffer_size', 10000)
        
        # Get tokenizer configuration
        tokenizer_config = self.dataset_config.get('tokenizer', {})
        self.tokenizer = self.model_manager.get_tokenizer(
            tokenizer_type=tokenizer_config.get('type', 'wordpiece'),
            model_name=tokenizer_config.get('model_name', 'bert-base-uncased')
        )
    
    def setup(
        self,
        split: str,
        batch_size: int,
        max_sequence_length: Optional[int] = None
    ) -> tf.data.Dataset:
        """Set up and preprocess a dataset split."""
        self.logger.info(f"Setting up dataset {self.dataset_name} - {split}")
        
        # Load raw data
        raw_data = self._load_raw_data(split)
        
        # Preprocess data
        processed_data = self._preprocess_data(
            raw_data,
            max_sequence_length or self.dataset_config.get('max_sequence_length', 512)
        )
        
        # Create TFRecords
        tfrecord_path = self._write_tfrecords(processed_data, split)
        
        # Create TF Dataset
        dataset = self._create_dataset(tfrecord_path, batch_size)
        
        return dataset
    
    def _load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load raw data for a given split."""
        data_path = Path(self.dataset_config['data_dir']) / f"{split}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} examples from {data_path}")
        return data
    
    def _preprocess_data(
        self,
        data: List[Dict[str, Any]],
        max_sequence_length: int
    ) -> List[Dict[str, Any]]:
        """Preprocess raw data."""
        processed = []
        for example in data:
            try:
                processed_example = self._preprocess_example(example, max_sequence_length)
                if processed_example:
                    processed.append(processed_example)
            except Exception as e:
                self.logger.warning(f"Error preprocessing example: {e}")
                continue
        
        self.logger.info(f"Preprocessed {len(processed)} examples")
        return processed
    
    def _preprocess_example(
        self,
        example: Dict[str, Any],
        max_sequence_length: int
    ) -> Optional[Dict[str, Any]]:
        """Preprocess a single example."""
        # Get text field from config
        text_field = self.dataset_config.get('text_field', 'text')
        if text_field not in example:
            self.logger.warning(f"Text field {text_field} not found in example")
            return None
        
        # Tokenize text
        encoding = self.tokenizer(
            example[text_field],
            max_length=max_sequence_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Convert to dict and add labels if present
        processed = {
            'input_ids': encoding['input_ids'].numpy()[0],
            'attention_mask': encoding['attention_mask'].numpy()[0]
        }
        
        # Add labels if present
        label_field = self.dataset_config.get('label_field')
        if label_field and label_field in example:
            processed['labels'] = example[label_field]
        
        return processed
    
    def _write_tfrecords(
        self,
        data: List[Dict[str, Any]],
        split: str
    ) -> Path:
        """Write data to sharded TFRecord files."""
        output_path = self.output_dir / f"{self.dataset_name}_{split}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate examples per shard
        num_examples = len(data)
        examples_per_shard = math.ceil(num_examples / self.num_shards)
        
        # Shuffle data if configured
        if self.shuffle_buffer_size > 0:
            import random
            random.shuffle(data)
        
        # Write shards
        for shard_idx in range(self.num_shards):
            start_idx = shard_idx * examples_per_shard
            end_idx = min(start_idx + examples_per_shard, num_examples)
            
            shard_path = output_path / f"shard_{shard_idx:05d}.tfrecord"
            self._write_shard(data[start_idx:end_idx], shard_path)
            
            self.logger.info(f"Written shard {shard_idx + 1}/{self.num_shards}")
        
        return output_path
    
    def _write_shard(self, data: List[Dict[str, Any]], path: Path):
        """Write a single TFRecord shard."""
        with tf.io.TFRecordWriter(str(path)) as writer:
            for example in data:
                tf_example = self._create_tf_example(example)
                writer.write(tf_example.SerializeToString())
    
    def _create_tf_example(self, example: Dict[str, Any]) -> tf.train.Example:
        """Create a TF Example from a preprocessed example."""
        feature = {
            'input_ids': tf.train.Feature(
                int64_list=tf.train.Int64List(value=example['input_ids'])
            ),
            'attention_mask': tf.train.Feature(
                int64_list=tf.train.Int64List(value=example['attention_mask'])
            )
        }
        
        if 'labels' in example:
            feature['labels'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[example['labels']])
            )
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def _create_dataset(
        self,
        tfrecord_path: Path,
        batch_size: int
    ) -> tf.data.Dataset:
        """Create a TF Dataset from TFRecord files."""
        feature_description = {
            'input_ids': tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            ),
            'attention_mask': tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            )
        }
        
        if self.dataset_config.get('label_field'):
            feature_description['labels'] = tf.io.FixedLenFeature([], tf.int64)
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(
                example_proto, feature_description
            )
        
        # Create dataset from TFRecord files
        file_pattern = str(tfrecord_path / "shard_*.tfrecord")
        dataset = tf.data.Dataset.list_files(file_pattern)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Parse and batch
        dataset = dataset.map(
            _parse_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset for training.")
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="Name of the dataset to prepare (must be in data_config.yaml).")
    parser.add_argument("--split", type=str, default="train",
                      help="Dataset split to prepare (e.g., train, validation, test).")
    parser.add_argument("--config_path", type=str, required=True,
                      help="Path to the data configuration YAML file.")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="GCS path to save the prepared dataset.")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting dataset preparation")
    
    try:
        # Instantiate the DatasetSetup class
        data_setup = DatasetSetup(
            config_manager=ConfigurationManager(args.config_path),
            model_manager=ModelManager(),
            dataset_name=args.dataset_name
        )
        
        # Prepare the dataset
        logger.info("Preparing dataset...")
        dataset = data_setup.setup(split=args.split, batch_size=args.batch_size)
        
        # Force dataset processing by taking one batch
        logger.info("Verifying dataset preparation...")
        for _ in dataset.take(1):
            pass
        
        logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 