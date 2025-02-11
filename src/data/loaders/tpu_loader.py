import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Optional, Dict, Any
import os

class TPUDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or xm.xrt_world_size()
        self.rank = rank or xm.get_ordinal()
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        
        # Create distributed sampler
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=shuffle,
            seed=seed
        )
        
        # Create data loader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )
        
        # Create TPU parallel loader
        self.parallel_loader = pl.MpDeviceLoader(
            self.loader,
            xm.xla_device()
        )
    
    def __iter__(self):
        return iter(self.parallel_loader)
    
    def __len__(self):
        return len(self.loader)
    
    def set_epoch(self, epoch: int):
        """Set the epoch for the sampler."""
        self.sampler.set_epoch(epoch)

class TPUPrefetchLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        self.base_loader = TPUDataLoader(
            dataset,
            batch_size,
            num_replicas,
            rank,
            shuffle,
            seed,
            num_workers,
            drop_last,
            pin_memory
        )
        self.prefetch_factor = prefetch_factor
        
        # Create prefetch queue
        self.queue = []
    
    def __iter__(self):
        iterator = iter(self.base_loader)
        
        # Prefetch initial batches
        for _ in range(self.prefetch_factor):
            try:
                batch = next(iterator)
                self.queue.append(batch)
            except StopIteration:
                break
        
        # Main iteration loop
        try:
            while self.queue:
                # Return the next batch
                batch = self.queue.pop(0)
                
                # Prefetch next batch
                try:
                    next_batch = next(iterator)
                    self.queue.append(next_batch)
                except StopIteration:
                    pass
                
                yield batch
        
        finally:
            # Clear queue
            self.queue.clear()
    
    def __len__(self):
        return len(self.base_loader)
    
    def set_epoch(self, epoch: int):
        """Set the epoch for the base loader."""
        self.base_loader.set_epoch(epoch)

def create_tpu_data_loader(
    dataset: Dataset,
    batch_size: int,
    is_training: bool = True,
    use_prefetch: bool = True,
    **kwargs
) -> TPUDataLoader:
    """Factory function to create appropriate TPU data loader."""
    if use_prefetch and is_training:
        return TPUPrefetchLoader(
            dataset,
            batch_size,
            shuffle=is_training,
            **kwargs
        )
    else:
        return TPUDataLoader(
            dataset,
            batch_size,
            shuffle=is_training,
            **kwargs
        ) 