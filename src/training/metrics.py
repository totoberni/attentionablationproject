import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_name: str, value: float, count: int = 1):
        """Update running average of a metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0.0
            self.counts[metric_name] = 0
        
        self.metrics[metric_name] += value * count
        self.counts[metric_name] += count
    
    def average(self, metric_name: str) -> float:
        """Get running average of a metric."""
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name] / max(1, self.counts[metric_name])
    
    def result(self) -> Dict[str, float]:
        """Get all metrics averages."""
        return {name: self.average(name) for name in self.metrics}

class ClassificationMetrics:
    def __init__(
        self,
        num_classes: int,
        average: str = 'macro'
    ):
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ):
        """Update metrics with new predictions."""
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all classification metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average=self.average
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        # ROC AUC and PR AUC (for multi-class, using one-vs-rest)
        roc_auc = roc_auc_score(
            targets,
            probabilities,
            multi_class='ovr',
            average=self.average
        )
        pr_auc = average_precision_score(
            targets,
            probabilities,
            average=self.average
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm
        }

class SequenceMetrics:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
        self.lengths = []
    
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """Update metrics with new predictions."""
        preds = torch.argmax(logits, dim=-1)
        
        # Remove padding tokens
        if attention_mask is not None:
            preds = [p[m.bool()] for p, m in zip(preds, attention_mask)]
            targets = [t[m.bool()] for t, m in zip(targets, attention_mask)]
        else:
            # Remove pad tokens
            preds = [p[t != self.pad_token_id] for p, t in zip(preds, targets)]
            targets = [t[t != self.pad_token_id] for t in targets]
        
        self.predictions.extend([p.cpu().numpy() for p in preds])
        self.targets.extend([t.cpu().numpy() for t in targets])
        self.lengths.extend([len(p) for p in preds])
    
    def compute(self) -> Dict[str, float]:
        """Compute sequence-level metrics."""
        # Exact match
        exact_matches = [
            np.array_equal(p, t)
            for p, t in zip(self.predictions, self.targets)
        ]
        exact_match_ratio = np.mean(exact_matches)
        
        # Token-level accuracy
        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)
        token_accuracy = accuracy_score(all_targets, all_preds)
        
        # Sequence lengths
        avg_length = np.mean(self.lengths)
        
        return {
            'exact_match': exact_match_ratio,
            'token_accuracy': token_accuracy,
            'average_length': avg_length
        }

class MLMMetrics:
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_predictions = 0
        self.correct_predictions = 0
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[torch.Tensor] = None
    ):
        """Update MLM metrics."""
        predictions = torch.argmax(logits, dim=-1)
        mask = (labels != self.ignore_index)
        
        self.total_predictions += mask.sum().item()
        self.correct_predictions += ((predictions == labels) & mask).sum().item()
        
        if loss is not None:
            self.total_loss += loss.item()
            self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute MLM metrics."""
        accuracy = self.correct_predictions / max(1, self.total_predictions)
        perplexity = torch.exp(torch.tensor(self.total_loss / max(1, self.num_batches)))
        
        return {
            'mlm_accuracy': accuracy,
            'perplexity': perplexity.item()
        } 