from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import yaml
import argparse
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)

class DatasetSetup:
    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        split: str = "train",
        output_dir: Optional[str] = None
    ):
        # Load config from file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config["datasets"][dataset_name]
        self.split = split
        self.output_dir = output_dir
        
        # Initialize preprocessing components
        self.text_cleaner = self._setup_text_cleaner()
    
    def _setup_text_cleaner(self):
        """Setup text cleaner based on configuration."""
        preprocessing_config = self.config["preprocessing"]
        
        def clean_text(text: tf.Tensor) -> tf.Tensor:
            # Only perform text cleaning, no tokenization
            if preprocessing_config["remove_html"]:
                text = tf.strings.regex_replace(text, '<[^>]*>', ' ')
            if preprocessing_config["normalize_unicode"]:
                text = tf.strings.unicode_normalize(text, 'NFKC')
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
        
        # Save to GCS if output directory is specified
        if self.output_dir:
            self._save_to_gcs(dataset)
        
        return dataset
    
    def _load_dataset(self) -> tf.data.Dataset:
        """Load dataset from source."""
        try:
            dataset = tfds.load(
                self.config["name"],
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
    
    def _save_to_gcs(self, dataset: tf.data.Dataset) -> None:
        """Save preprocessed dataset to GCS."""
        if not self.output_dir.startswith('gs://'):
            raise ValueError("Output directory must be a GCS path")
        
        # Create TFRecord options
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        
        # Ensure the output directory exists
        tf.io.gfile.makedirs(self.output_dir)
        
        # Save dataset to TFRecord
        output_path = f"{self.output_dir}/{self.split}.tfrecord"
        logger.info(f"Saving dataset to {output_path}")
        
        writer = tf.data.experimental.TFRecordWriter(
            output_path,
            compression_type='GZIP'
        )
        writer.write(dataset)
        
        logger.info(f"Dataset saved successfully to {output_path}")

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
            config_path=args.config_path,
            dataset_name=args.dataset_name,
            split=args.split,
            output_dir=args.output_dir
        )
        
        # Prepare the dataset
        logger.info("Preparing dataset...")
        dataset = data_setup.prepare_dataset(batch_size=args.batch_size)
        
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