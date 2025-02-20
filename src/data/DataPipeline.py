from typing import Dict, Optional, List, Union, Tuple, Any
from datasets import DatasetDict
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

from src.data.core import (
    ConfigurationManager,
    DependencyManager,
    ModelManager,
    DatasetSetup,
    TaskType,
    ModelInput,
    ModelTarget,
    verify_alignment,
    verify_task_alignment,
    verify_token_alignment,
    create_sharded_dataset
)
from src.data.preprocessing.Inputs import InputProcessor
from src.data.preprocessing.Targets import TargetHandler

logger = logging.getLogger(__name__)

class DataPipeline:
    """Coordinates data processing components and manages the overall pipeline."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        dependency_manager: Optional[DependencyManager] = None,
        model_manager: Optional[ModelManager] = None,
        dataset_setup: Optional[DatasetSetup] = None,
        input_processor: Optional[InputProcessor] = None,
        target_handler: Optional[TargetHandler] = None
    ):
        """Initialize the data pipeline with injected dependencies."""
        self.config_manager = config_manager
        
        # Initialize or use injected components
        self.dependency_manager = dependency_manager or DependencyManager(config_manager)
        self.model_manager = model_manager or ModelManager(config_manager)
        self.dataset_setup = dataset_setup or DatasetSetup(config_manager)
        self.input_processor = input_processor or InputProcessor(config_manager, self.model_manager)
        self.target_handler = target_handler or TargetHandler(config_manager, self.model_manager)
        
        # Load pipeline configuration
        self.pipeline_config = self.config_manager.get_config('pipeline_config')
        
        # Initialize processing state
        self.processing_state = {
            'current_dataset': None,
            'current_model_type': None,
            'processed_tasks': set(),
            'alignment_verified': False
        }
    
    def process_data(
        self, 
        data_source: Union[DatasetDict, List[str]],
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer",
        batch_size: int = 32,
        num_shards: int = 1,
        shuffle_buffer_size: int = 10000
    ) -> Union[DatasetDict, Dict[str, tf.Tensor]]:
        """Process data with unified processing logic and optional sharding."""
        try:
            # Update processing state
            self._update_processing_state(dataset_name, model_type)
            
            # Initialize dependencies if needed
            if isinstance(data_source, DatasetDict):
                self.dependency_manager.install_dependencies()
            
            # Process based on input type
            if isinstance(data_source, list):
                return self._process_single_batch(
                    data_source,
                    dataset_name,
                    tasks,
                    model_type,
                    batch_size
                )
            else:
                return self._process_dataset_splits(
                    data_source,
                    dataset_name,
                    tasks,
                    model_type,
                    batch_size,
                    num_shards,
                    shuffle_buffer_size
                )
                
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise
    
    def _update_processing_state(self, dataset_name: str, model_type: str) -> None:
        """Update the current processing state."""
        if (
            dataset_name != self.processing_state['current_dataset'] or
            model_type != self.processing_state['current_model_type']
        ):
            self.processing_state.update({
                'current_dataset': dataset_name,
                'current_model_type': model_type,
                'processed_tasks': set(),
                'alignment_verified': False
            })
    
    def _process_single_batch(
        self,
        texts: List[str],
        dataset_name: str,
        tasks: Optional[List[TaskType]],
        model_type: str,
        batch_size: int
    ) -> Dict[str, tf.Tensor]:
        """Process a single batch with enhanced error handling and metadata."""
        logger.info(f"Processing batch for {model_type} model...")
        
        try:
            # Process in smaller batches if needed
            if len(texts) > batch_size:
                return self._process_large_batch(
                    texts,
                    dataset_name,
                    tasks,
                    model_type,
                    batch_size
                )
            
            # Forward processing
            model_inputs = self.input_processor.process_inputs(
                texts=texts,
                dataset_name=dataset_name,
                model_type=model_type
            )
            
            model_targets = self.target_handler.generate_targets(
                texts=texts,
                dataset_name=dataset_name,
                tasks=tasks,
                model_type=model_type
            )
            
            # Verify alignment
            self._verify_bidirectional_alignment(
                model_inputs.metadata,
                model_targets.metadata,
                dataset_name,
                model_type
            )
            
            # Update processing state
            self.processing_state['alignment_verified'] = True
            self.processing_state['processed_tasks'].update(
                model_targets.metadata.get('task_info', {}).keys()
            )
            
            # Create metadata
            metadata = self._create_batch_metadata(
                dataset_name,
                model_type,
                len(texts)
            )
            
            # Combine and return tensors
            return {
                **model_inputs.to_tensors(),
                **model_targets.to_tensors(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _process_large_batch(
        self,
        texts: List[str],
        dataset_name: str,
        tasks: Optional[List[TaskType]],
        model_type: str,
        batch_size: int
    ) -> Dict[str, tf.Tensor]:
        """Process a large batch in smaller chunks."""
        logger.info(f"Processing large batch of size {len(texts)} in chunks of {batch_size}")
        
        # Process in chunks
        processed_chunks = []
        for i in tqdm(range(0, len(texts), batch_size)):
            chunk = texts[i:i + batch_size]
            processed_chunk = self._process_single_batch(
                chunk,
                dataset_name,
                tasks,
                model_type,
                batch_size
            )
            processed_chunks.append(processed_chunk)
        
        # Combine chunks
        combined = {}
        for key in processed_chunks[0].keys():
            if key == 'metadata':
                combined[key] = self._merge_chunk_metadata(
                    [chunk[key] for chunk in processed_chunks]
                )
            else:
                combined[key] = np.concatenate(
                    [chunk[key] for chunk in processed_chunks],
                    axis=0
                )
        
        return combined
    
    def _process_dataset_splits(
        self,
        dataset: DatasetDict,
        dataset_name: str,
        tasks: Optional[List[TaskType]],
        model_type: str,
        batch_size: int,
        num_shards: int,
        shuffle_buffer_size: int
    ) -> DatasetDict:
        """Process dataset splits with sharding support."""
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split...")
            
            # Create sharded dataset
            sharded_data = create_sharded_dataset(
                split_data['text'],
                num_shards,
                shuffle_buffer_size
            )
            
            # Process each shard
            processed_shards = []
            for shard in sharded_data.batch(batch_size):
                processed_shard = self._process_single_batch(
                    shard.numpy().tolist(),
                    dataset_name,
                    tasks,
                    model_type,
                    batch_size
                )
                processed_shards.append(processed_shard)
            
            # Combine shards
            processed_features = {}
            for key in processed_shards[0].keys():
                if key == 'metadata':
                    processed_features[key] = self._merge_chunk_metadata(
                        [shard[key] for shard in processed_shards]
                    )
                else:
                    processed_features[key] = np.concatenate(
                        [shard[key] for shard in processed_shards],
                        axis=0
                    )
            
            # Update dataset
            processed_dataset[split_name] = split_data.add_columns(processed_features)
            
            logger.info(f"Successfully processed {split_name} split")
        
        return DatasetDict(processed_dataset)
    
    def _verify_bidirectional_alignment(
        self,
        input_metadata: Dict,
        target_metadata: Dict,
        dataset_name: str,
        model_type: str
    ) -> None:
        """Verify bidirectional alignment with enhanced error reporting."""
        logger.info("Performing bidirectional alignment verification...")
        
        try:
            # Forward verification
            input_to_target = verify_alignment(
                input_metadata,
                target_metadata,
                direction="forward"
            )
            
            # Backward verification
            target_to_input = verify_alignment(
                target_metadata,
                input_metadata,
                direction="backward"
            )
            
            # Task-specific verification
            tasks_aligned = self._verify_task_alignments(
                input_metadata.get('task_inputs', {}),
                target_metadata.get('task_info', {})
            )
            
            if not (input_to_target and target_to_input and tasks_aligned):
                raise ValueError(
                    f"Alignment verification failed:\n"
                    f"Dataset: {dataset_name}\n"
                    f"Model type: {model_type}\n"
                    f"Input→Target: {input_to_target}\n"
                    f"Target→Input: {target_to_input}\n"
                    f"Tasks aligned: {tasks_aligned}"
                )
                
        except Exception as e:
            logger.error(f"Alignment verification failed: {str(e)}")
            raise
    
    def _verify_task_alignments(
        self,
        input_tasks: Dict[str, Dict],
        target_tasks: Dict[str, Dict]
    ) -> bool:
        """Verify alignment of all task-specific data."""
        try:
            for task in input_tasks:
                if task in target_tasks:
                    if not verify_task_alignment(
                        input_tasks[task],
                        target_tasks[task]
                    ):
                        logger.error(f"Task alignment failed for {task}")
                        return False
            return True
            
        except Exception as e:
            logger.error(f"Error during task alignment verification: {str(e)}")
            return False
    
    def _create_batch_metadata(
        self,
        dataset_name: str,
        model_type: str,
        batch_size: int
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for processed batch."""
        return {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'model_type': model_type,
                'batch_size': batch_size,
                'processed_tasks': list(self.processing_state['processed_tasks']),
                'alignment_verified': self.processing_state['alignment_verified']
            },
            'pipeline_version': self.pipeline_config.get('version', '1.0.0'),
            'component_versions': {
                'input_processor': self.input_processor.__class__.__version__,
                'target_handler': self.target_handler.__class__.__version__,
                'model_manager': self.model_manager.__class__.__version__
            }
        }
    
    def _merge_chunk_metadata(self, chunk_metadata: List[Dict]) -> Dict[str, Any]:
        """Merge metadata from multiple chunks."""
        base_metadata = chunk_metadata[0].copy()
        
        # Update processing info
        base_metadata['processing_info'].update({
            'num_chunks': len(chunk_metadata),
            'total_samples': sum(
                m['processing_info']['batch_size']
                for m in chunk_metadata
            ),
            'chunk_timestamps': [
                m['processing_info']['timestamp']
                for m in chunk_metadata
            ]
        })
        
        return base_metadata

    def reconstruct_data(
        self,
        processed_data: Union[Dict[str, tf.Tensor], DatasetDict],
        reconstruction_type: str = "both"
    ) -> Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """Reconstruct original text from processed data."""
        if isinstance(processed_data, dict):
            return self._reconstruct_single_batch(processed_data, reconstruction_type)
        else:
            return {
                split: self._reconstruct_single_batch(split_data, reconstruction_type)
                for split, split_data in processed_data.items()
            }
    
    def _reconstruct_single_batch(
        self,
        batch: Dict[str, tf.Tensor],
        reconstruction_type: str
    ) -> Dict[str, List[str]]:
        """Reconstruct original text from a single processed batch."""
        result = {}
        
        try:
            if reconstruction_type in ["input", "both"]:
                result["input_texts"] = self.input_processor.reconstruct_text(
                    batch["input_ids"].numpy(),
                    batch["input_metadata"]
                )
            
            if reconstruction_type in ["target", "both"]:
                result["target_texts"] = self.target_handler.reconstruct_text(
                    batch["reconstruction_targets"].numpy(),
                    batch["target_metadata"]
                )
            
            # Verify reconstruction alignment
            if reconstruction_type == "both":
                self._verify_reconstruction_alignment(
                    result.get("input_texts", []),
                    result.get("target_texts", [])
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during batch reconstruction: {str(e)}")
            return {}
    
    def _verify_reconstruction_alignment(
        self,
        input_texts: List[str],
        target_texts: List[str]
    ) -> None:
        """Verify alignment between reconstructed input and target texts."""
        if len(input_texts) != len(target_texts):
            logger.error("Reconstruction length mismatch")
            return
        
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            if input_text != target_text:
                logger.warning(f"Reconstruction mismatch at index {i}:")
                logger.warning(f"Input:  {input_text}")
                logger.warning(f"Target: {target_text}")

    def save_to_tfrecord(self, data: Dict[str, tf.Tensor], output_path: str) -> None:
        """Save processed data to TFRecord format on GCS."""
        def _create_example(features: Dict[str, tf.Tensor]) -> tf.train.Example:
            feature_dict = {}
            
            for key, value in features.items():
                if isinstance(value, (int, bool)):
                    feature = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[value])
                    )
                elif isinstance(value, float):
                    feature = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[value])
                    )
                elif isinstance(value, str):
                    feature = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value.encode()])
                    )
                elif isinstance(value, (tf.Tensor, np.ndarray)):
                    if value.dtype in [tf.int32, tf.int64, bool]:
                        feature = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=value.flatten())
                        )
                    elif value.dtype in [tf.float32, tf.float64]:
                        feature = tf.train.Feature(
                            float_list=tf.train.FloatList(value=value.flatten())
                        )
                    else:
                        feature = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
                        )
                else:
                    continue
                
                feature_dict[key] = feature
            
            return tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not tf.io.gfile.exists(output_dir):
            tf.io.gfile.makedirs(output_dir)
        
        # Write TFRecord file
        with tf.io.TFRecordWriter(output_path) as writer:
            for example in data:
                tf_example = _create_example(example)
                writer.write(tf_example.SerializeToString())
        
        logger.info(f"Saved TFRecord file to: {output_path}")

    def _verify_gcs_path(self, path: str) -> bool:
        """Verify that a GCS path exists and is accessible."""
        try:
            return tf.io.gfile.exists(path)
        except:
            return False

    def _get_dataset_from_gcs(self, gcs_path: str) -> tf.data.Dataset:
        """Load a dataset from GCS."""
        if not self._verify_gcs_path(gcs_path):
            raise FileNotFoundError(f"GCS path not found: {gcs_path}")
        
        file_pattern = os.path.join(gcs_path, "*.tfrecord")
        filenames = tf.io.gfile.glob(file_pattern)
        
        if not filenames:
            raise ValueError(f"No TFRecord files found in {gcs_path}")
        
        return tf.data.TFRecordDataset(filenames)
