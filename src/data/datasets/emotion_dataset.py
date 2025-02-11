import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer

class EmotionDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        split: str = "train",
        text_column: str = "text",
        label_column: str = "label",
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Load dataset
        self.dataset = load_dataset("dar-ai/emotion", split=split)
        self.text_column = text_column
        self.label_column = label_column
        
        # Get label mapping
        self.label2id = {label: idx for idx, label in enumerate(self.dataset.features[label_column].names)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item[self.text_column],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get label
        label = torch.tensor(self.label2id[item[self.label_column]])
        
        # Prepare MLM inputs if needed
        if self.mlm_probability > 0:
            mlm_inputs = self._prepare_mlm_inputs(encoding["input_ids"].squeeze(0))
            encoding["mlm_input_ids"] = mlm_inputs["input_ids"]
            encoding["mlm_labels"] = mlm_inputs["labels"]
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add label
        encoding["labels"] = label
        
        return encoding
    
    def _prepare_mlm_inputs(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prepare inputs for masked language modeling."""
        labels = input_ids.clone()
        
        # Create probability mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self._get_special_tokens_mask(labels)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for unmasked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100
        
        # Mask tokens
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # Replace some masked tokens with random words
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {"input_ids": input_ids, "labels": labels}
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a mask for special tokens."""
        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id
        }
        return torch.tensor([1 if x in special_tokens else 0 for x in input_ids], dtype=torch.bool)

    @property
    def num_labels(self) -> int:
        return len(self.label2id) 