import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Callable
import wandb
from tqdm import tqdm
import os
import logging
from ..data.loaders.tpu_loader import TPUDataLoader

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: TPUDataLoader,
        val_loader: Optional[TPUDataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[Callable] = None,
        num_epochs: int = 40,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        use_wandb: bool = True,
        early_stopping_patience: int = 3
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.device = device or xm.xla_device()
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize logging
        self._setup_logging()
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(epoch)
            self._log_metrics(train_metrics, "train", epoch)
            
            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)
                self._log_metrics(val_metrics, "val", epoch)
                
                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info("Early stopping triggered")
                        break
            
            # Save regular checkpoint
            self._save_checkpoint(epoch, train_metrics['loss'])
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch['labels'])
                
                # Backward pass
                loss.backward()
                xm.optimizer_step(self.optimizer)
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': total_loss / num_batches}
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch['labels'])
                
                # Update metrics
                total_loss += loss.item()
        
        return {'loss': total_loss / num_batches}
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        xm.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            xm.save(checkpoint, best_path)
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str, epoch: int):
        """Log metrics to console and wandb."""
        # Console logging
        metrics_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch+1} {phase} metrics: {metrics_str}")
        
        # Wandb logging
        if self.use_wandb:
            wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=epoch)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler('logs/training.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler) 