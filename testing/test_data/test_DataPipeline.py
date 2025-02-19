import unittest
import os
from pathlib import Path
import tensorflow as tf
from datasets import DatasetDict, Dataset
import numpy as np
from typing import Dict, List

from src.data.DataPipeline import DataPipeline
from src.data.core import TaskType

class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all test methods."""
        # Get the project root directory
        cls.project_root = Path(__file__).parent.parent.parent
        cls.config_dir = cls.project_root / "config"
        
        # Initialize the pipeline
        cls.pipeline = DataPipeline(str(cls.config_dir))
        
        # Sample texts for batch testing
        cls.sample_texts = [
            "This is a test sentence for emotion analysis.",
            "Another sample text to process through the pipeline.",
            "Testing the data pipeline with various inputs.",
            "Making sure everything works as expected."
        ]
    
    def setUp(self):
        """Set up test fixtures that will be used for each test method."""
        # Create a mock dataset for testing
        self.mock_dataset = DatasetDict({
            'train': Dataset.from_dict({
                'text': self.sample_texts[:2],
                'label': [0, 1]
            }),
            'validation': Dataset.from_dict({
                'text': self.sample_texts[2:],
                'label': [1, 0]
            })
        })
    
    def _print_processing_steps(self, name: str, texts: List[str], processed_data: Dict, reconstructed: Dict):
        """Helper to print the processing steps for better visualization."""
        print(f"\n{'='*20} {name} {'='*20}")
        print("\nOriginal Texts:")
        for i, text in enumerate(texts):
            print(f"{i+1}. {text}")
        
        print("\nEncoded Data:")
        for key, value in processed_data.items():
            if isinstance(value, (tf.Tensor, np.ndarray)):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"{key}: {type(value)}")
        
        print("\nReconstructed Texts:")
        for key, texts in reconstructed.items():
            print(f"\n{key}:")
            for i, text in enumerate(texts):
                print(f"{i+1}. {text}")
        
        print("\n" + "="*50)
    
    def test_batch_processing(self):
        """Test processing a batch of texts."""
        print("\nTesting Batch Processing")
        
        # Process batch
        processed_batch = self.pipeline.process_data(
            self.sample_texts,
            dataset_name="emotion",
            tasks=[TaskType.SENTIMENT, TaskType.MLM],
            model_type="transformer",
            is_batch=True
        )
        
        # Reconstruct batch
        reconstructed = self.pipeline.reconstruct_data(processed_batch)
        
        # Print processing steps
        self._print_processing_steps(
            "Batch Processing",
            self.sample_texts,
            processed_batch,
            reconstructed
        )
        
        # Assertions
        self.assertIn('input_ids', processed_batch)
        self.assertIn('sequence_mask', processed_batch)
        self.assertIn('input_texts', reconstructed)
        self.assertEqual(len(reconstructed['input_texts']), len(self.sample_texts))
    
    def test_emotion_dataset(self):
        """Test processing the emotion dataset."""
        print("\nTesting Emotion Dataset Processing")
        
        # Process dataset
        processed_dataset = self.pipeline.process_data(
            self.mock_dataset,
            dataset_name="emotion",
            tasks=[TaskType.SENTIMENT, TaskType.MLM],
            model_type="transformer",
            is_batch=False
        )
        
        # Test each split
        for split_name, split_data in processed_dataset.items():
            # Get original texts
            original_texts = self.mock_dataset[split_name]['text']
            
            # Convert split data to dict for reconstruction
            split_dict = {
                key: value for key, value in split_data.items()
                if isinstance(value, (tf.Tensor, np.ndarray, dict))
            }
            
            # Reconstruct
            reconstructed = self.pipeline.reconstruct_data(split_dict)
            
            # Print processing steps
            self._print_processing_steps(
                f"Emotion Dataset - {split_name} split",
                original_texts,
                split_dict,
                reconstructed
            )
            
            # Assertions
            self.assertIn('input_ids', split_dict)
            self.assertIn('sequence_mask', split_dict)
            self.assertIn('input_texts', reconstructed)
            self.assertEqual(len(reconstructed['input_texts']), len(original_texts))
    
    def test_gutenberg_dataset(self):
        """Test processing the Gutenberg dataset."""
        print("\nTesting Gutenberg Dataset Processing")
        
        # Process dataset
        processed_dataset = self.pipeline.process_data(
            self.mock_dataset,
            dataset_name="gutenberg",
            tasks=[TaskType.MLM, TaskType.LMLM, TaskType.NSP],
            model_type="transformer",
            is_batch=False
        )
        
        # Test each split
        for split_name, split_data in processed_dataset.items():
            # Get original texts
            original_texts = self.mock_dataset[split_name]['text']
            
            # Convert split data to dict for reconstruction
            split_dict = {
                key: value for key, value in split_data.items()
                if isinstance(value, (tf.Tensor, np.ndarray, dict))
            }
            
            # Reconstruct
            reconstructed = self.pipeline.reconstruct_data(split_dict)
            
            # Print processing steps
            self._print_processing_steps(
                f"Gutenberg Dataset - {split_name} split",
                original_texts,
                split_dict,
                reconstructed
            )
            
            # Assertions
            self.assertIn('input_ids', split_dict)
            self.assertIn('sequence_mask', split_dict)
            self.assertIn('input_texts', reconstructed)
            self.assertEqual(len(reconstructed['input_texts']), len(original_texts))

if __name__ == '__main__':
    unittest.main() 