import tensorflow as tf
from typing import Dict, List, Optional, Union, Callable
import wandb
from tqdm import tqdm
import os
import logging
from ..data.loaders.tpu_loader import TPUDataLoader
import json

class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        train_loader: TPUDataLoader,
        val_loader: Optional[TPUDataLoader] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        criterion: Optional[Callable] = None,
        num_epochs: int = 40,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        use_wandb: bool = True,
        early_stopping_patience: int = 3
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=2e-5)
        self.scheduler = scheduler
        self.criterion = criterion or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.num_epochs = num_epochs
        self.device = device or '/TPU:0'
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize logging
        self._setup_logging()
        
        # Set up TPU strategy
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        self.strategy = tf.distribute.TPUStrategy(resolver)
        
        # Move model to TPU strategy scope
        with self.strategy.scope():
            self.model = model
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                with tf.GradientTape() as tape:
                    outputs = self.model(batch, training=True)
                    loss = self.criterion(batch['labels'], outputs)
                
                # Compute gradients and apply
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # Update metrics
                total_loss += loss.numpy()
                
                # Update progress bar
                pbar.update(1)
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({'loss': float(loss.numpy())})
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            outputs = self.model(batch, training=False)
            loss = self.criterion(batch['labels'], outputs)
            total_loss += loss.numpy()
        
        return {'loss': total_loss / num_batches}
    
    def train(self) -> Dict[str, float]:
        """Train the model for the specified number of epochs."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate epoch
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)
                current_val_loss = val_metrics['loss']
                
                # Early stopping check
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics if self.val_loader else None)
        
        return {'best_val_loss': best_val_loss}
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.h5"
        )
        self.model.save_weights(checkpoint_path)
        
        # Save metrics
        metrics_path = os.path.join(
            self.checkpoint_dir,
            f"metrics_epoch_{epoch}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics to wandb and console."""
        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_metrics:
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        if self.use_wandb:
            wandb.log(metrics, step=epoch)
        
        # Log to console
        metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - {metrics_str}") 