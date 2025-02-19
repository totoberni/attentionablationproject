from typing import Dict, Optional, List, Union, Tuple
from datasets import DatasetDict
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime

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
    
    def prepare_dataset(
        self, 
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> DatasetDict:
        """Prepare a dataset with all necessary processing steps."""
        # Initialize dependencies if needed
        self.dependency_manager.install_dependencies()
        
        # Load and setup dataset
        dataset = self.dataset_setup.initialize_dataset(dataset_name)
        
        # Process dataset
        processed_dataset = self._process_dataset(dataset, dataset_name, tasks, model_type)
        
        return processed_dataset
    
    def _process_dataset(
        self, 
        dataset: DatasetDict, 
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> DatasetDict:
        """Apply all processing steps to the dataset with bidirectional alignment validation."""
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split for {model_type} model...")
            raw_texts = split_data['text']
            
            # Forward direction: Raw Text -> Targets -> Validate with Inputs
            logger.info("Generating targets and validating alignment with inputs...")
            model_targets: ModelTarget = self.target_handler.generate_targets(
                raw_texts,
                dataset_name,
                tasks,
                model_type
            )
            
            model_inputs: ModelInput = self.input_processor.process_inputs(
                raw_texts,
                dataset_name,
                model_type
            )
            
            # Validate target-to-input alignment
            if not self.verify_alignment(model_targets.metadata, model_inputs.metadata, "target_to_input"):
                raise ValueError(
                    f"Target-to-Input alignment validation failed in {split_name} split\n"
                    f"Model type: {model_type}\n"
                    f"Dataset: {dataset_name}"
                )
            
            # Reverse direction: Raw Text -> Inputs -> Validate with Targets
            logger.info("Validating input-to-target alignment...")
            if not self.verify_alignment(model_inputs.metadata, model_targets.metadata, "input_to_target"):
                raise ValueError(
                    f"Input-to-Target alignment validation failed in {split_name} split\n"
                    f"Model type: {model_type}\n"
                    f"Dataset: {dataset_name}"
                )
            
            # Store alignment verification metadata
            alignment_metadata = {
                'alignment_verified': True,
                'timestamp': datetime.now().isoformat(),
                'split': split_name,
                'model_type': model_type,
                'dataset': dataset_name,
                'processed_tasks': list(self.input_processor.processed_inputs),
                'alignment_validations': {
                    'target_to_input': True,
                    'input_to_target': True,
                    'validation_timestamp': datetime.now().isoformat()
                }
            }
            
            # Convert to tensors and combine
            input_tensors = model_inputs.to_tensors()
            target_tensors = model_targets.to_tensors()
            
            # Update dataset with processed features
            processed_features = {
                **input_tensors,
                **target_tensors,
                'metadata': alignment_metadata
            }
            dataset[split_name] = split_data.add_columns(processed_features)
            
            logger.info(f"Successfully processed {split_name} split with bidirectional alignment validation")
        
        return dataset
    
    def process_batch(
        self, 
        texts: List[str], 
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> Dict[str, tf.Tensor]:
        """Process a single batch of texts."""
        logger.info(f"Processing batch for {model_type} model...")
        
        # Process inputs
        model_inputs: ModelInput = self.input_processor.process_inputs(
            texts=texts,
            dataset_name=dataset_name,
            model_type=model_type
        )
        
        # Generate targets
        model_targets: ModelTarget = self.target_handler.generate_targets(
            texts=texts, 
            dataset_name=dataset_name,
            tasks=tasks,
            model_type=model_type
        )
        
        # Verify alignment
        if not self.verify_alignment(model_inputs.metadata, model_targets.metadata):
            raise ValueError(
                f"Input and target processing misalignment detected in batch\n"
                f"Model type: {model_type}\n"
                f"Dataset: {dataset_name}"
            )
        
        # Store alignment verification in metadata
        alignment_metadata = {
            'alignment_verified': True,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'dataset': dataset_name,
            'processed_tasks': list(self.input_processor.processed_inputs)
        }
        
        # Combine tensors
        processed_batch = {
            **model_inputs.to_tensors(),
            **model_targets.to_tensors(),
            'metadata': alignment_metadata
        }
        
        logger.info(f"Successfully processed batch with {len(processed_batch)} features")
        return processed_batch
    
    def verify_alignment(
        self,
        source_metadata: Dict,
        target_metadata: Dict,
        direction: str
    ) -> bool:
        """Verify alignment between source and target processing with direction tracking."""
        logger.info(f"Verifying {direction} alignment...")
        
        try:
            # Verify using both processors for extra robustness
            if direction == "target_to_input":
                input_aligned = self.input_processor.verify_alignment(
                    target_metadata['tokenization_info'],
                    source_metadata
                )
                target_aligned = self.target_handler.verify_alignment(
                    source_metadata,
                    target_metadata
                )
            else:  # input_to_target
                input_aligned = self.input_processor.verify_alignment(
                    source_metadata,
                    target_metadata['tokenization_info']
                )
                target_aligned = self.target_handler.verify_alignment(
                    target_metadata,
                    source_metadata
                )
            
            if not input_aligned or not target_aligned:
                logger.error(f"{direction} alignment verification failed")
                logger.error("Source metadata: %s", source_metadata)
                logger.error("Target metadata: %s", target_metadata)
                return False
            
            # Additional checks for task-specific alignment
            if 'task_inputs' in source_metadata:
                for task, task_inputs in source_metadata['task_inputs'].items():
                    if task in target_metadata.get('task_info', {}):
                        if not self._verify_task_alignment(
                            task,
                            task_inputs,
                            target_metadata['task_info'][task]
                        ):
                            logger.error(f"Task alignment failed for {task} in {direction}")
                            return False
            
            logger.info(f"{direction} alignment verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Error during {direction} alignment verification: {str(e)}")
            return False
    
    def _verify_task_alignment(
        self,
        task: str,
        input_task_data: Dict,
        target_task_data: Dict
    ) -> bool:
        """Verify alignment for task-specific data."""
        try:
            if task in ["mlm", "lmlm"]:
                # Check mask positions match
                input_masks = input_task_data.get('masked_positions')
                target_masks = target_task_data.get('masked_positions')
                if input_masks is not None and target_masks is not None:
                    if not np.array_equal(input_masks, target_masks):
                        logger.error(f"Mask position mismatch for {task}")
                        return False
            
            elif task == "nsp":
                # Check pair information matches
                input_pairs = input_task_data.get('nsp_labels')
                target_pairs = target_task_data.get('nsp_labels')
                if input_pairs is not None and target_pairs is not None:
                    if not np.array_equal(input_pairs, target_pairs):
                        logger.error("NSP label mismatch")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error during task alignment verification: {str(e)}")
            return False
    
    def reconstruct_batch(
        self,
        batch: Dict[str, tf.Tensor],
        source: str = "both"
    ) -> Dict[str, List[str]]:
        """Reconstruct original text from processed batch."""
        result = {}
        
        try:
            if source in ["input", "both"]:
                result["input_texts"] = self.input_processor.reconstruct_text(
                    batch["input_ids"].numpy(),
                    batch["input_metadata"]
                )
            
            if source in ["target", "both"]:
                result["target_texts"] = self.target_handler.reconstruct_text(
                    batch["reconstruction_targets"].numpy(),
                    batch["target_metadata"]
                )
            
            # Verify reconstruction alignment
            if source == "both" and len(result.get("input_texts", [])) == len(result.get("target_texts", [])):
                for i, (input_text, target_text) in enumerate(zip(result["input_texts"], result["target_texts"])):
                    if input_text != target_text:
                        logger.warning(f"Reconstruction mismatch at index {i}:")
                        logger.warning(f"Input:  {input_text}")
                        logger.warning(f"Target: {target_text}")
            
            return result
        except Exception as e:
            logger.error(f"Error during batch reconstruction: {str(e)}")
            return {}
