import os
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer
import sentencepiece as spm
import tempfile

class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        special_tokens: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Set default special tokens if not provided
        self.special_tokens = {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "cls": "[CLS]",
            "sep": "[SEP]",
            "mask": "[MASK]"
        }
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # Initialize tokenizer
        self.sp_model = None
        self._token_to_id = {}
        self._id_to_token = {}
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to the vocabulary."""
        for idx, (_, token) in enumerate(self.special_tokens.items()):
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
    
    def train(
        self,
        files: Union[str, List[str]],
        output_dir: str,
        vocab_size: Optional[int] = None
    ):
        """Train the SentencePiece tokenizer."""
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # Create temporary training config
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(f"""
                input={','.join(files) if isinstance(files, list) else files}
                model_prefix={os.path.join(output_dir, 'tokenizer')}
                vocab_size={vocab_size}
                model_type={self.model_type}
                pad_id={self._token_to_id[self.special_tokens['pad']]}
                unk_id={self._token_to_id[self.special_tokens['unk']]}
                bos_id=-1
                eos_id=-1
                user_defined_symbols={','.join([self.special_tokens['cls'], self.special_tokens['sep'], self.special_tokens['mask']])}
            """)
            config_path = f.name
        
        # Train tokenizer
        spm.SentencePieceTrainer.Train(config_path)
        os.unlink(config_path)
        
        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(os.path.join(output_dir, "tokenizer.model"))
        
        # Update vocabulary
        for i in range(self.sp_model.GetPieceSize()):
            token = self.sp_model.IdToPiece(i)
            if token not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> Union[List[int], List[List[int]]]:
        """Encode text to token ids."""
        if isinstance(text, str):
            text = [text]
        
        encoded = []
        for t in text:
            if add_special_tokens:
                t = f"{self.special_tokens['cls']} {t} {self.special_tokens['sep']}"
            
            if self.sp_model:
                ids = self.sp_model.EncodeAsIds(t)
            else:
                # Fallback to basic tokenization if model not trained
                tokens = t.split()
                ids = [self._token_to_id.get(token, self._token_to_id[self.special_tokens['unk']]) 
                       for token in tokens]
            
            encoded.append(ids)
        
        return encoded[0] if len(text) == 1 else encoded
    
    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """Decode token ids to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if not isinstance(token_ids[0], list):
            token_ids = [token_ids]
        
        decoded = []
        for ids in token_ids:
            if self.sp_model:
                text = self.sp_model.DecodeIds(ids)
            else:
                # Fallback to basic detokenization
                tokens = [self._id_to_token.get(id_, self.special_tokens['unk']) for id_ in ids]
                if skip_special_tokens:
                    tokens = [t for t in tokens if t not in self.special_tokens.values()]
                text = " ".join(tokens)
            
            decoded.append(text)
        
        return decoded[0] if len(token_ids) == 1 else decoded
    
    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        return self._token_to_id.copy()
    
    def save_pretrained(self, save_directory: str):
        """Save the tokenizer to a directory."""
        if self.sp_model:
            self.sp_model.Save(os.path.join(save_directory, "tokenizer.model"))
    
    @classmethod
    def from_pretrained(cls, directory: str, **kwargs):
        """Load a pretrained tokenizer from a directory."""
        tokenizer = cls(**kwargs)
        model_path = os.path.join(directory, "tokenizer.model")
        if os.path.exists(model_path):
            tokenizer.sp_model = spm.SentencePieceProcessor()
            tokenizer.sp_model.Load(model_path)
        return tokenizer 