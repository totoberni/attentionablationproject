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
from src.utils import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configuration and logging
    config_manager = ConfigurationManager(args.config)
    logger = logging.Logger(
        name="evaluation",
        log_dir=args.output,
        use_tensorboard=True,
        use_wandb=True
    )
    
    # Setup dataset
    dataset = DatasetSetup(config_manager, args.dataset)
    input_processor = InputProcessor(config_manager, args.dataset)
    target_generator = TargetGenerator(config_manager, args.dataset)
    
    # Initialize model
    model_registry = ModelRegistry()
    model_manager = ModelManager(model_registry)
    
    # Load model with checkpoint
    model_config = config_manager.get_config("model_config")
    model = model_manager.create_model(
        args.model,
        model_config["models"][args.model],
        weights_path=args.checkpoint
    )
    
    # Setup trainer for evaluation
    trainer = Trainer(
        model=model,
        val_loader=dataset.prepare_dataset(batch_size=32, shuffle=False, split="test"),
        checkpoint_dir=args.output
    )
    
    # Run evaluation
    metrics = trainer._validate_epoch(epoch=0)
    logger.log_metrics(metrics, prefix="test")

if __name__ == "__main__":
    main() 