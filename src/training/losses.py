import tensorflow as tf
from typing import Optional, Dict, Union

class MLMLoss(tf.keras.layers.Layer):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Compute masked language modeling loss."""
        # Reshape logits and labels for loss computation
        logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
        labels = tf.reshape(labels, [-1])
        
        # Create mask for ignored indices
        mask = tf.not_equal(labels, self.ignore_index)
        labels = tf.boolean_mask(labels, mask)
        logits = tf.boolean_mask(logits, mask)
        
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
        )

class NSPLoss(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Compute next sentence prediction loss."""
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                labels, logits, from_logits=True
            )
        )

class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def call(self, embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Compute contrastive loss using cosine similarity."""
        # Normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        
        # Compute similarity matrix
        similarity = tf.matmul(embeddings, embeddings, transpose_b=True) / self.temperature
        
        # Create labels matrix
        labels_matrix = tf.equal(
            tf.expand_dims(labels, 0),
            tf.expand_dims(labels, 1)
        )
        labels_matrix = tf.cast(labels_matrix, tf.float32)
        
        # Mask out self-similarity
        mask = tf.eye(tf.shape(similarity)[0])
        similarity = similarity * (1 - mask)
        
        # Compute loss
        exp_sim = tf.exp(similarity)
        log_prob = similarity - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
        loss = -(log_prob * labels_matrix) / tf.reduce_sum(labels_matrix, axis=1)
        
        return tf.reduce_mean(loss)

class MultiTaskLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        tasks: Dict[str, tf.keras.layers.Layer],
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.tasks = tasks
        self.weights = weights or {task: 1.0 for task in tasks}
    
    def call(
        self,
        outputs: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
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

class FocalLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        alpha: Optional[tf.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def call(self, logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """Compute focal loss for imbalanced classification."""
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            targets, logits, from_logits=True
        )
        pt = tf.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = tf.gather(self.alpha, tf.cast(targets, tf.int32))
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return tf.reduce_mean(focal_loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(focal_loss)
        return focal_loss

class HierarchicalLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        hierarchy: Dict[int, List[int]],
        base_criterion: Optional[tf.keras.layers.Layer] = None
    ):
        super().__init__()
        self.hierarchy = hierarchy
        self.base_criterion = base_criterion or tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
    
    def call(
        self,
        logits: tf.Tensor,
        targets: tf.Tensor,
        level_weights: Optional[List[float]] = None
    ) -> tf.Tensor:
        """Compute hierarchical classification loss."""
        if level_weights is None:
            level_weights = [1.0] * len(self.hierarchy)
        
        total_loss = 0.0
        
        for level, weight in enumerate(level_weights):
            # Get classes for current level
            level_classes = self.hierarchy[level]
            
            # Select relevant logits and map targets
            level_logits = tf.gather(logits, level_classes, axis=1)
            level_targets = self._map_targets(targets, level)
            
            # Compute loss for current level
            level_loss = self.base_criterion(level_targets, level_logits)
            total_loss += weight * level_loss
        
        return total_loss
    
    def _map_targets(self, targets: tf.Tensor, level: int) -> tf.Tensor:
        """Map targets to their corresponding level in the hierarchy."""
        level_classes = self.hierarchy[level]
        mapping = {cls: idx for idx, cls in enumerate(level_classes)}
        return tf.convert_to_tensor(
            [mapping[t.numpy()] for t in tf.unstack(targets)],
            dtype=tf.int32
        ) 