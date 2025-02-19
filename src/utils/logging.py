import logging
import os
import sys
from typing import Optional, Dict, Any
import json
import yaml
from datetime import datetime
import wandb
import tensorflow as tf

class Logger:
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_file_logging()
        
        # Initialize trackers
        if use_tensorboard:
            self.writer = tf.summary.create_file_writer(self.log_dir)
        
        if use_wandb:
            wandb.init(
                project=name,
                config=config,
                dir=self.log_dir
            )
    
    def _setup_file_logging(self):
        """Setup file and console logging."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "run.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics to all active trackers."""
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to file
        metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Log to tensorboard
        if self.use_tensorboard:
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        # Save to file
        params_path = os.path.join(self.log_dir, "hyperparameters.yaml")
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        
        # Log to tensorboard
        if self.use_tensorboard:
            for name, value in params.items():
                tf.summary.scalar(name, value)
        
        # Log to wandb
        if self.use_wandb:
            wandb.config.update(params)
    
    def log_model_summary(self, model_summary: str):
        """Log model architecture summary."""
        # Save to file
        summary_path = os.path.join(self.log_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(model_summary)
        
        # Log to wandb
        if self.use_wandb:
            wandb.save(summary_path)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        # Save to file
        config_path = os.path.join(self.log_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Log to wandb
        if self.use_wandb:
            wandb.save(config_path)
    
    def log_artifact(self, artifact_path: str, artifact_type: str):
        """Log an artifact file."""
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=os.path.basename(artifact_path),
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
    
    def close(self):
        """Close all logging connections."""
        if self.use_tensorboard:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()

class MetricLogger:
    def __init__(self, delimiter: str = "  "):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)
    
    def __str__(self):
        """Format metric values as string."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

class SmoothedValue:
    def __init__(
        self,
        window_size: int = 20,
        fmt: Optional[str] = None
    ):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"
    
    def update(self, value: float, n: int = 1):
        """Update value and maintain moving window."""
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
    
    @property
    def median(self) -> float:
        """Calculate median of current window."""
        return float(np.median(self.deque))
    
    @property
    def avg(self) -> float:
        """Calculate average of current window."""
        return float(np.mean(self.deque))
    
    @property
    def global_avg(self) -> float:
        """Calculate global average."""
        return self.total / max(self.count, 1)
    
    def __str__(self):
        """Format values according to format string."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg
        ) 