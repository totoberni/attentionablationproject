#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.management import ConfigurationManager, ModelRegistry, ModelManager
from src.core import DatasetSetup, InputProcessor, TargetGenerator
from src.training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configuration
    config_manager = ConfigurationManager(args.config)
    
    # Setup dataset
    dataset = DatasetSetup(config_manager, args.dataset)
    input_processor = InputProcessor(config_manager, args.dataset)
    target_generator = TargetGenerator(config_manager, args.dataset)
    
    # Initialize model
    model_registry = ModelRegistry()
    model_manager = ModelManager(model_registry)
    
    # Get model configuration
    model_config = config_manager.get_config("model_config")
    model = model_manager.create_model(args.model, model_config["models"][args.model])
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=dataset.prepare_dataset(batch_size=32, shuffle=True),
        val_loader=dataset.prepare_dataset(batch_size=32, shuffle=False, split="validation"),
        checkpoint_dir=args.output
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 