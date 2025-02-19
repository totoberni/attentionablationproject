from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
from .core import DatasetSetup
from .preprocessing.inputs import InputProcessor
from .preprocessing.targets import TargetGenerator, TargetProcessor

class DataPipeline:
    def __init__(
        self,
        config_manager,
        dataset_name: str,
        split: str = "train"
    ):
        self.config_manager = config_manager
        self.dataset_name = dataset_name
        self.split = split
        
        # Initialize components
        self.dataset_setup = DatasetSetup(config_manager, dataset_name, split)
        self.input_processor = InputProcessor(config_manager, dataset_name)
        self.target_generator = TargetGenerator(config_manager, dataset_name)
        self.target_processor = TargetProcessor(config_manager, dataset_name)
    
    def prepare_dataset(
        self,
        batch_size: int,
        tasks: Optional[List[str]] = None,
        shuffle: bool = True,
        cache: bool = True
    ) -> tf.data.Dataset:
        """Prepare dataset with specified tasks."""
        # Get base dataset
        dataset = self.dataset_setup.prepare_dataset(
            batch_size=batch_size,
            shuffle=shuffle,
            cache=cache
        )
        
        # If no tasks specified, use all tasks from config
        if tasks is None:
            tasks = self.dataset_setup.dataset_config["tasks"].keys()
        
        # Process each task
        def process_batch(batch):
            features = {}
            texts = batch["texts"]
            
            # Process inputs for each task
            for task in tasks:
                task_inputs = self.input_processor.prepare_inputs(texts, task)
                features.update({
                    f"{task}_{k}": v for k, v in task_inputs.items()
                })
                
                # Generate and process targets if needed
                if "labels" not in batch:
                    targets = self.target_generator.generate_targets(texts, task)
                    processed_targets = self.target_processor.process_targets(targets, task)
                    features.update({
                        f"{task}_{k}": v for k, v in processed_targets.items()
                    })
            
            return features
        
        # Apply processing to dataset
        return dataset.map(
            process_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    def process_batch(
        self,
        texts: Union[str, List[str]],
        tasks: Optional[List[str]] = None
    ) -> Dict[str, tf.Tensor]:
        """Process a single batch of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # If no tasks specified, use all tasks from config
        if tasks is None:
            tasks = self.dataset_setup.dataset_config["tasks"].keys()
        
        features = {}
        
        # Process inputs for each task
        for task in tasks:
            task_inputs = self.input_processor.prepare_inputs(texts, task)
            features.update({
                f"{task}_{k}": v for k, v in task_inputs.items()
            })
            
            # Generate and process targets
            targets = self.target_generator.generate_targets(texts, task)
            processed_targets = self.target_processor.process_targets(targets, task)
            features.update({
                f"{task}_{k}": v for k, v in processed_targets.items()
            })
        
        return features
