import torch
import numpy as np
from typing import Dict, List, Optional, Union
import random
import re

class TextAugmentation:
    def __init__(
        self,
        swap_prob: float = 0.1,
        delete_prob: float = 0.1,
        replace_prob: float = 0.1,
        max_swap_distance: int = 3
    ):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.replace_prob = replace_prob
        self.max_swap_distance = max_swap_distance
    
    def __call__(self, text: str) -> str:
        """Apply random augmentations to the text."""
        words = text.split()
        
        # Random word swap
        if random.random() < self.swap_prob:
            words = self._swap_words(words)
        
        # Random word deletion
        if random.random() < self.delete_prob:
            words = self._delete_words(words)
        
        # Random word replacement
        if random.random() < self.replace_prob:
            words = self._replace_words(words)
        
        return " ".join(words)
    
    def _swap_words(self, words: List[str]) -> List[str]:
        """Randomly swap nearby words."""
        if len(words) < 2:
            return words
        
        words = words.copy()
        for i in range(len(words)):
            if random.random() < self.swap_prob:
                max_dist = min(self.max_swap_distance, len(words) - i - 1)
                if max_dist > 0:
                    j = i + random.randint(1, max_dist)
                    words[i], words[j] = words[j], words[i]
        
        return words
    
    def _delete_words(self, words: List[str]) -> List[str]:
        """Randomly delete words."""
        if len(words) < 2:
            return words
        
        return [w for w in words if random.random() > self.delete_prob]
    
    def _replace_words(self, words: List[str]) -> List[str]:
        """Replace words with similar length random strings."""
        words = words.copy()
        for i, word in enumerate(words):
            if random.random() < self.replace_prob:
                words[i] = self._generate_random_word(len(word))
        
        return words
    
    @staticmethod
    def _generate_random_word(length: int) -> str:
        """Generate a random word of given length."""
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

class TokenizerTransform:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        padding: bool = True,
        truncation: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize text and convert to model inputs."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length' if self.padding else False,
            truncation=self.truncation,
            return_tensors='pt'
        )

class TextCleaner:
    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        lowercase: bool = True
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.number_pattern = re.compile(r'\d+')
        self.punct_pattern = re.compile(r'[^\w\s]')
    
    def __call__(self, text: str) -> str:
        """Clean the input text."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        if self.remove_punctuation:
            text = self.punct_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

class CompositeTransform:
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms
    
    def __call__(self, text: str) -> Union[str, Dict[str, torch.Tensor]]:
        """Apply multiple transforms in sequence."""
        for transform in self.transforms:
            text = transform(text)
        return text 