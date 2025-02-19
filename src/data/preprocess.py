import tensorflow as tf
import tensorflow_io as tfio
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
import argparse
import yaml
from google.cloud import storage
from pathlib import Path

from src.data.core import (
    ConfigurationManager,
    DependencyManager,
    ModelManager,
    DatasetSetup,
    TaskType
)
from src.data.DataPipeline import DataPipeline

logger = logging.getLogger(__name__)

class GCSDataManager:
    """Manages data synchronization with Google Cloud Storage."""
    
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def sync_directory_to_gcs(self, local_path: str, gcs_path: str) -> None:
        """Synchronizes a local directory to GCS."""
        local_path = Path(local_path)
        for local_file in local_path.rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path)
                blob_path = str(Path(gcs_path) / relative_path)
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(str(local_file))
                logger.info(f"Uploaded {local_file} to gs://{self.bucket.name}/{blob_path}")
    
    def sync_directory_from_gcs(self, gcs_path: str, local_path: str) -> None:
        """Synchronizes a GCS directory to local storage."""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        
        blobs = self.bucket.list_blobs(prefix=gcs_path)
        for blob in blobs:
            relative_path = Path(blob.name).relative_to(gcs_path)
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
            logger.info(f"Downloaded gs://{self.bucket.name}/{blob.name} to {local_file}")

def setup_logging(log_dir: str) -> None:
    """Sets up logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocess.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_dir: str) -> Dict:
    """Loads and merges configuration files."""
    config_dir = Path(config_dir)
    config = {}
    
    for config_file in config_dir.glob('*.yaml'):
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))
    
    return config

def preprocess_dataset(
    pipeline: DataPipeline,
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    config: Dict
) -> None:
    """Preprocesses a dataset and saves it to GCS."""
    # Load dataset configuration
    dataset_config = config.get('datasets', {}).get(dataset_name, {})
    if not dataset_config:
        raise ValueError(f"No configuration found for dataset {dataset_name}")
    
    # Get enabled tasks
    tasks = [TaskType(task) for task in dataset_config.get('enabled_tasks', [])]
    
    # Load and preprocess data
    data = pipeline.process_data(
        data_source=input_dir,
        dataset_name=dataset_name,
        tasks=tasks,
        model_type=dataset_config.get('model_type', 'transformer'),
        is_batch=False
    )
    
    # Save processed data
    output_path = Path(output_dir) / dataset_name
    tf.io.gfile.makedirs(str(output_path))
    
    for split_name, split_data in data.items():
        split_path = output_path / f"{split_name}.tfrecord"
        pipeline.save_to_tfrecord(split_data, str(split_path))
        logger.info(f"Saved {split_name} split to {split_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for TPU training")
    parser.add_argument("--input-dir", required=True, help="GCS path to raw data")
    parser.add_argument("--output-dir", required=True, help="GCS path to store processed data")
    parser.add_argument("--config-dir", required=True, help="Path to configuration directory")
    parser.add_argument("--tokenizer-path", required=True, help="GCS path to tokenizer models")
    parser.add_argument("--cache-dir", required=True, help="GCS path for caching")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.cache_dir)
    logger.info("Starting preprocessing pipeline")
    
    try:
        # Load configuration
        config = load_config(args.config_dir)
        
        # Initialize GCS data manager
        gcs_manager = GCSDataManager(config['storage']['bucket'])
        
        # Initialize data pipeline
        pipeline = DataPipeline(args.config_dir)
        
        # Process each dataset
        for dataset_name in config['datasets']:
            logger.info(f"Processing dataset: {dataset_name}")
            try:
                preprocess_dataset(
                    pipeline=pipeline,
                    dataset_name=dataset_name,
                    input_dir=f"{args.input_dir}/{dataset_name}",
                    output_dir=args.output_dir,
                    config=config
                )
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
        
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 