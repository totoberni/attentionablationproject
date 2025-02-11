import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer
import random

class GutenbergDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
        text_column: str = "chosen",
        summary_column: str = "summary",
        mlm_probability: float = 0.15,
        nsp_probability: float = 0.5
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.nsp_probability = nsp_probability
        
        # Load dataset
        self.dataset = load_dataset("gutemberg2", split=split)
        self.text_column = text_column
        self.summary_column = summary_column
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Get text segments for NSP
        if random.random() < self.nsp_probability:
            # Get consecutive segments (IsNext)
            text_a, text_b = self._get_consecutive_segments(item[self.text_column])
            nsp_label = 1
        else:
            # Get random segments (NotNext)
            text_a = self._get_segment(item[self.text_column])
            text_b = self._get_segment(self.dataset[random.randint(0, len(self.dataset)-1)][self.text_column])
            nsp_label = 0
        
        # Tokenize segments
        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare MLM inputs
        if self.mlm_probability > 0:
            mlm_inputs = self._prepare_mlm_inputs(encoding["input_ids"].squeeze(0))
            encoding["mlm_input_ids"] = mlm_inputs["input_ids"]
            encoding["mlm_labels"] = mlm_inputs["labels"]
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add NSP label
        encoding["nsp_labels"] = torch.tensor(nsp_label)
        
        return encoding
    
    def _get_consecutive_segments(self, text: str) -> Tuple[str, str]:
        """Split text into two consecutive segments."""
        sentences = text.split(". ")
        if len(sentences) < 2:
            return text, text  # Fallback for very short texts
        
        split_idx = len(sentences) // 2
        text_a = ". ".join(sentences[:split_idx]) + "."
        text_b = ". ".join(sentences[split_idx:])
        
        return text_a, text_b
    
    def _get_segment(self, text: str) -> str:
        """Get a random segment from text."""
        sentences = text.split(". ")
        if len(sentences) < 2:
            return text
        
        start_idx = random.randint(0, len(sentences)-2)
        end_idx = min(start_idx + random.randint(1, 3), len(sentences))
        
        return ". ".join(sentences[start_idx:end_idx]) + "."
    
    def _prepare_mlm_inputs(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prepare inputs for masked language modeling."""
        labels = input_ids.clone()
        
        # Create probability mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self._get_special_tokens_mask(labels)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for unmasked tokens to -100
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