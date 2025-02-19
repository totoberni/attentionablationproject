from typing import Dict, Optional, List, Union, Tuple
from datasets import DatasetDict
import logging
import tensorflow as tf
import numpy as np

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
        """Apply all processing steps to the dataset."""
        for split_name, split_data in dataset.items():
            # Process inputs
            model_inputs: ModelInput = self.input_processor.process_inputs(
                split_data['text'],
                dataset_name,
                model_type
            )
            
            # Generate targets
            model_targets: ModelTarget = self.target_handler.generate_targets(
                split_data['text'],
                dataset_name,
                tasks,
                model_type
            )
            
            # Convert to tensors and combine
            input_tensors = model_inputs.to_tensors()
            target_tensors = model_targets.to_tensors()
            
            # Update dataset with processed features
            processed_features = {**input_tensors, **target_tensors}
            dataset[split_name] = split_data.add_columns(processed_features)
        
        return dataset
    
    def process_batch(
        self, 
        texts: List[str], 
        dataset_name: str,
        tasks: Optional[List[TaskType]] = None,
        model_type: str = "transformer"
    ) -> Dict[str, tf.Tensor]:
        """Process a single batch of texts."""
        model_inputs: ModelInput = self.input_processor.process_inputs(
            texts,
            dataset_name,
            model_type
        )
        model_targets: ModelTarget = self.target_handler.generate_targets(
            texts, 
            dataset_name,
            tasks,
            model_type
        )
        
        return {
            **model_inputs.to_tensors(),
            **model_targets.to_tensors()
        }
