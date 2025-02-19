from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Any
from enum import Enum, auto
import numpy as np
import tensorflow as tf

class TaskType(Enum):
    MLM = auto()
    LMLM = auto()
    NSP = auto()
    SENTIMENT = auto()
    NER = auto()
    POS = auto()
    DISCOURSE = auto()
    CONTRASTIVE = auto()

@dataclass
class ModelInput:
    """Container for model input tensors with support for both transformer and static embeddings."""
    input_ids: np.ndarray
    sequence_mask: np.ndarray
    decoder_input: Optional[np.ndarray] = None  # For structural compatibility
    cls_token_mask: Optional[np.ndarray] = None  # For transformer models
    sep_token_mask: Optional[np.ndarray] = None  # For transformer models
    
    def to_tensors(self) -> Dict[str, tf.Tensor]:
        """Convert numpy arrays to TensorFlow tensors."""
        tensors = {
            'input_ids': tf.convert_to_tensor(self.input_ids, dtype=tf.int32),
            'sequence_mask': tf.convert_to_tensor(self.sequence_mask, dtype=tf.int32)
        }
        
        optional_tensors = {
            'decoder_input': self.decoder_input,
            'cls_token_mask': self.cls_token_mask,
            'sep_token_mask': self.sep_token_mask
        }
        
        tensors.update({
            key: tf.convert_to_tensor(value, dtype=tf.int32)
            for key, value in optional_tensors.items()
            if value is not None
        })
        
        return tensors

@dataclass
class TaskLabels:
    """Container for task-specific labels and masks."""
    task_type: TaskType
    labels: np.ndarray
    mask: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelTarget:
    """Container for model target tensors with task-specific labels."""
    reconstruction_targets: np.ndarray
    sequence_mask: np.ndarray
    task_labels: Dict[TaskType, TaskLabels] = field(default_factory=dict)
    metadata: Optional[Dict[str, np.ndarray]] = None
    
    def add_task_labels(self, task_type: TaskType, labels: np.ndarray, 
                       mask: Optional[np.ndarray] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add labels for a specific task."""
        self.task_labels[task_type] = TaskLabels(
            task_type=task_type,
            labels=labels,
            mask=mask,
            metadata=metadata
        )
    
    def to_tensors(self) -> Dict[str, tf.Tensor]:
        """Convert all arrays to TensorFlow tensors."""
        tensors = {
            'reconstruction_targets': tf.convert_to_tensor(self.reconstruction_targets, dtype=tf.int32),
            'sequence_mask': tf.convert_to_tensor(self.sequence_mask, dtype=tf.int32)
        }
        
        # Convert task-specific labels
        for task_type, task_labels in self.task_labels.items():
            task_name = task_type.name.lower()
            tensors[f'{task_name}_labels'] = tf.convert_to_tensor(task_labels.labels, dtype=tf.int32)
            
            if task_labels.mask is not None:
                tensors[f'{task_name}_mask'] = tf.convert_to_tensor(task_labels.mask, dtype=tf.int32)
            
            if task_labels.metadata:
                for meta_key, meta_value in task_labels.metadata.items():
                    tensors[f'{task_name}_{meta_key}'] = tf.convert_to_tensor(meta_value)
        
        # Add general metadata if present
        if self.metadata:
            for key, value in self.metadata.items():
                tensors[f'metadata_{key}'] = tf.convert_to_tensor(value)
        
        return tensors 