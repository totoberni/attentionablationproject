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
from src.models.probing import structural, information
from src.utils import visualization, logging

def parse_args():
    parser = argparse.ArgumentParser(description="Probing analysis script")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--probe-type", type=str, required=True, choices=["structural", "information"],
                      help="Type of probing analysis")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configuration and logging
    config_manager = ConfigurationManager(args.config)
    logger = logging.Logger(
        name="probing",
        log_dir=args.output,
        use_tensorboard=True,
        use_wandb=True
    )
    
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
    
    # Run probing analysis
    if args.probe_type == "structural":
        probe = structural.StructuralProbe(model)
        results = probe.analyze()
        visualization.plot_structural_analysis(results, save_path=args.output)
    else:  # information
        probe = information.InformationProbe(model)
        results = probe.analyze()
        visualization.plot_information_flow(results, save_path=args.output)
    
    # Log results
    logger.log_metrics(results, prefix=args.probe_type)

if __name__ == "__main__":
    main() 