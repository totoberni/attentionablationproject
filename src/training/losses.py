import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union

class MLMLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute masked language modeling loss."""
        # Reshape logits and labels for loss computation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        return self.loss_fn(logits, labels)

class NSPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute next sentence prediction loss."""
        return self.loss_fn(logits.squeeze(-1), labels.float())

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using cosine similarity."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create labels matrix
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float()
        
        # Mask out self-similarity
        mask = torch.eye(similarity.size(0), device=similarity.device)
        similarity = similarity * (1 - mask)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(log_prob * labels_matrix).sum(dim=1) / labels_matrix.sum(dim=1)
        
        return loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        tasks: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.tasks = tasks
        self.weights = weights or {task: 1.0 for task in tasks}
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted sum of task-specific losses."""
        losses = {}
        total_loss = 0.0
        
        for task_name, loss_fn in self.tasks.items():
            if task_name in outputs and task_name in targets:
                task_loss = loss_fn(outputs[task_name], targets[task_name])
                losses[task_name] = task_loss
                total_loss += self.weights[task_name] * task_loss
        
        losses['total'] = total_loss
        return losses

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for imbalanced classification."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class HierarchicalLoss(nn.Module):
    def __init__(
        self,
        hierarchy: Dict[int, List[int]],
        base_criterion: nn.Module = nn.CrossEntropyLoss()
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.base_criterion = base_criterion
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        level_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute hierarchical classification loss."""
        if level_weights is None:
            level_weights = [1.0] * len(self.hierarchy)
        
        total_loss = 0.0
        
        for level, weight in enumerate(level_weights):
            # Get classes for current level
            level_classes = self.hierarchy[level]
            
            # Select relevant logits and map targets
            level_logits = logits[:, level_classes]
            level_targets = self._map_targets(targets, level)
            
            # Compute loss for current level
            level_loss = self.base_criterion(level_logits, level_targets)
            total_loss += weight * level_loss
        
        return total_loss
    
    def _map_targets(self, targets: torch.Tensor, level: int) -> torch.Tensor:
        """Map targets to their corresponding level in the hierarchy."""
        level_classes = self.hierarchy[level]
        mapping = {cls: idx for idx, cls in enumerate(level_classes)}
        return torch.tensor([mapping[t.item()] for t in targets], device=targets.device) 