from typing import Dict, Optional, List, Union, Tuple
from datasets import DatasetDict
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
import os

from src.data.core import (
    ConfigurationManager,
    DependencyManager,
    ModelManager,
    DatasetSetup,
    TaskType,
    ModelInput,
    ModelTarget
)
from src.data.preprocessing.Inputs import InputProcessor
from src.data.preprocessing.Targets import TargetHandler

logger = logging.getLogger(__name__)

class DataPipeline:
    """Coordinates data processing components and manages the overall pipeline."""
    
    def __init__(self, config_dir: str):
        # Initialize core components
        self.config_manager = ConfigurationManager(config_dir)
        self.dependency_manager = DependencyManager(self.config_manager)
        self.model_manager = ModelManager(self.config_manager)
        
        # Initialize dataset setup
        self.dataset_setup = DatasetSetup(self.config_manager)
        
        # Initialize preprocessing components
        self.input_processor = InputProcessor(self.config_manager, self.model_manager)
        self.target_handler = TargetHandler(self.config_manager, self.model_manager)
    
    def process_data(
        self, 
        data_source: Union[DatasetDict, List[str]],
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer",
        is_batch: bool = False
    ) -> Union[DatasetDict, Dict[str, tf.Tensor]]:
        """Process data (either dataset or batch) with unified processing logic."""
        # Initialize dependencies if needed
        if not is_batch:
            self.dependency_manager.install_dependencies()
        
        if is_batch:
            return self._process_single_batch(data_source, dataset_name, tasks, model_type)
        else:
            return self._process_dataset_splits(data_source, dataset_name, tasks, model_type)
    
    def _process_single_batch(
        self,
        texts: List[str],
        dataset_name: str,
        tasks: Optional[List[TaskType]],
        model_type: str
    ) -> Dict[str, tf.Tensor]:
        """Process a single batch of texts with bidirectional verification."""
        logger.info(f"Processing batch for {model_type} model...")
        
        # Forward direction: Text → Input → Target verification
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
        
        # Verify bidirectional alignment
        self._verify_bidirectional_alignment(
            model_inputs.metadata,
            model_targets.metadata,
            dataset_name,
            model_type
        )
        
        # Store alignment verification in metadata
        alignment_metadata = {
            'alignment_verified': True,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'dataset': dataset_name,
            'processed_tasks': list(self.input_processor.processed_inputs),
            'verification_type': 'bidirectional'
        }
        
        # Combine tensors
        return {
            **model_inputs.to_tensors(),
            **model_targets.to_tensors(),
            'metadata': alignment_metadata
        }
    
    def _process_dataset_splits(
        self,
        dataset: DatasetDict,
        dataset_name: str,
        tasks: Optional[List[TaskType]],
        model_type: str
    ) -> DatasetDict:
        """Process all splits in a dataset with bidirectional verification."""
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split for {model_type} model...")
            raw_texts = split_data['text']
            
            # Process the split as a batch
            processed_features = self._process_single_batch(
                raw_texts,
                dataset_name,
                tasks,
                model_type
            )
            
            # Update dataset with processed features
            dataset[split_name] = split_data.add_columns(processed_features)
            
            logger.info(f"Successfully processed {split_name} split with bidirectional verification")
        
        return dataset
    
    def _verify_bidirectional_alignment(
        self,
        input_metadata: Dict,
        target_metadata: Dict,
        dataset_name: str,
        model_type: str
    ) -> None:
        """Verify bidirectional alignment between inputs and targets."""
        logger.info("Performing bidirectional alignment verification...")
        
        # Input → Target verification
        input_to_target = self._verify_alignment_direction(
            input_metadata,
            target_metadata,
            "input_to_target"
        )
        
        # Target → Input verification
        target_to_input = self._verify_alignment_direction(
            target_metadata,
            input_metadata,
            "target_to_input"
        )
        
        if not (input_to_target and target_to_input):
            raise ValueError(
                f"Bidirectional alignment verification failed\n"
                f"Model type: {model_type}\n"
                f"Dataset: {dataset_name}\n"
                f"Input→Target: {input_to_target}\n"
                f"Target→Input: {target_to_input}"
            )
    
    def _verify_alignment_direction(
        self,
        source_metadata: Dict,
        target_metadata: Dict,
        direction: str
    ) -> bool:
        """Verify alignment in a single direction with task verification."""
        try:
            # Base alignment verification
            if direction == "input_to_target":
                aligned = self.input_processor.verify_alignment(
                    source_metadata,
                    target_metadata.get('tokenization_info', {})
                )
            else:  # target_to_input
                aligned = self.target_handler.verify_alignment(
                    source_metadata,
                    target_metadata
                )
            
            if not aligned:
                logger.error(f"{direction} base alignment failed")
                return False
            
            # Task-specific alignment verification
            if 'task_inputs' in source_metadata:
                for task, task_data in source_metadata['task_inputs'].items():
                    if task in target_metadata.get('task_info', {}):
                        if not self._verify_task_data(
                            task,
                            task_data,
                            target_metadata['task_info'][task]
                        ):
                            logger.error(f"{direction} task alignment failed for {task}")
                            return False
            
            logger.info(f"{direction} alignment verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Error during {direction} alignment verification: {str(e)}")
            return False
    
    def _verify_task_data(
        self,
        task: str,
        source_data: Dict,
        target_data: Dict
    ) -> bool:
        """Verify alignment of task-specific data."""
        try:
            if task in ["mlm", "lmlm"]:
                return np.array_equal(
                    source_data.get('masked_positions'),
                    target_data.get('masked_positions')
                )
            
            elif task == "nsp":
                return np.array_equal(
                    source_data.get('nsp_labels'),
                    target_data.get('nsp_labels')
                )
            
            # Add more task-specific verifications as needed
            return True
            
        except Exception as e:
            logger.error(f"Error during task data verification for {task}: {str(e)}")
            return False
    
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
