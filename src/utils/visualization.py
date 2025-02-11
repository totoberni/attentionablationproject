import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb

class AttentionVisualizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer: int,
        head: int,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """Plot attention weights as a heatmap."""
        # Convert to numpy and squeeze if needed
        weights = attention_weights.squeeze().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            annot=True,
            fmt='.2f'
        )
        
        # Set title
        if title is None:
            title = f'Attention Weights (Layer {layer}, Head {head})'
        plt.title(title)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            try:
                wandb.log({title: wandb.Image(save_path)})
            except:
                pass
        else:
            plt.show()
    
    def visualize_attention_flow(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        save_path: Optional[str] = None
    ):
        """Visualize attention flow across layers."""
        num_layers = len(attention_weights)
        num_heads = attention_weights[0].size(0)
        
        # Create subplot grid
        fig, axes = plt.subplots(
            num_layers,
            num_heads,
            figsize=(4*num_heads, 4*num_layers)
        )
        
        for layer in range(num_layers):
            for head in range(num_heads):
                weights = attention_weights[layer][head].cpu().numpy()
                
                ax = axes[layer, head] if num_layers > 1 else axes[head]
                sns.heatmap(
                    weights,
                    xticklabels=tokens if layer == num_layers-1 else [],
                    yticklabels=tokens if head == 0 else [],
                    cmap='viridis',
                    ax=ax,
                    cbar=False
                )
                ax.set_title(f'Layer {layer+1}, Head {head+1}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            try:
                wandb.log({'Attention Flow': wandb.Image(save_path)})
            except:
                pass
        else:
            plt.show()

class EmbeddingVisualizer:
    @staticmethod
    def plot_embeddings(
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        method: str = 'tsne',
        perplexity: int = 30,
        n_components: int = 2,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """Visualize embeddings using dimensionality reduction."""
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        if labels is not None:
            labels = labels.cpu().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42
            )
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels if labels is not None else None,
            cmap='tab10' if labels is not None else None
        )
        
        if labels is not None:
            plt.colorbar(scatter)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'{method.upper()} Visualization of Embeddings')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            try:
                wandb.log({title or 'Embeddings': wandb.Image(save_path)})
            except:
                pass
        else:
            plt.show()

class LossVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """Update loss history."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
    
    def plot_losses(
        self,
        save_path: Optional[str] = None,
        title: str = 'Training and Validation Loss'
    ):
        """Plot loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss')
        
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            try:
                wandb.log({'Loss Curves': wandb.Image(save_path)})
            except:
                pass
        else:
            plt.show()

class ConfusionMatrixVisualizer:
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        save_path: Optional[str] = None,
        title: str = 'Confusion Matrix'
    ):
        """Plot confusion matrix."""
        if normalize:
            confusion_matrix = (
                confusion_matrix.astype('float') /
                confusion_matrix.sum(axis=1)[:, np.newaxis]
            )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if available
            try:
                wandb.log({'Confusion Matrix': wandb.Image(save_path)})
            except:
                pass
        else:
            plt.show() 