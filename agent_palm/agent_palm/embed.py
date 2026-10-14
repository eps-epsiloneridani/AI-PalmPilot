"""
ONNX Runtime embeddings for agent_palm
Lazy loading, HashEmbedder for tests, all-MiniLM-L6-v2 int8
"""

import os
import hashlib
import numpy as np
from typing import List, Optional, Sequence
from pathlib import Path


class OnnxEmbedder:
    """
    ONNX Runtime embedder for all-MiniLM-L6-v2
    Lazy loads model on first use to keep import fast.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        self._session = None
        self._tokenizer = None
        self._model_dir = model_dir
        self._model_loaded = False
    
    @property
    def model_dir(self) -> str:
        if self._model_dir:
            return self._model_dir
        
        # Default to MEMPALM_PATH/models or local models dir
        mempalm_path = os.environ.get("MEMPALM_PATH")
        if mempalm_path:
            return os.path.join(mempalm_path, "models")
        
        # Fallback to local models dir
        return os.path.join(os.path.dirname(__file__), "..", "models")
    
    def _ensure_loaded(self):
        """Lazy load model and tokenizer on first use"""
        if self._model_loaded:
            return
        
        model_dir = self.model_dir
        model_path = os.path.join(model_dir, "all-MiniLM-L6-v2_int8.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Embedding model not found at {model_path}. "
                f"Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 "
                f"and convert to ONNX int8, or run: python -m agent_palm.export_model"
            )
        
        # Load tokenizer
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(
                os.path.join(model_dir, "tokenizer.json")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Load ONNX session
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                os.path.join(model_dir, "all-MiniLM-L6-v2_int8.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=ort.SessionOptions()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
        
        self._model_loaded = True
    
    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        """
        Embed a batch of texts.
        Returns list of 384-dim float32 arrays (L2 normalized).
        """
        self._ensure_loaded()
        
        # Tokenize
        encodings = self._tokenizer.encode_batch(list(texts))
        
        # Pad to max length in batch
        max_len = max(len(e.ids) for e in encodings)
        input_ids = np.array([
            e.ids + [0] * (max_len - len(e.ids)) for e in encodings
        ], dtype=np.int64)
        
        attention_mask = np.array([
            e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encodings
        ], dtype=np.int64)
        
        # token_type_ids (all zeros for single-sequence inputs)
        token_type_ids = np.zeros_like(input_ids)
        
        # Run inference
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        
        # outputs[0] shape: (batch, seq_len, hidden_dim)
        embeddings = outputs[0]
        
        # Mean pooling with attention mask
        pooled = []
        for i, emb in enumerate(embeddings):
            mask = attention_mask[i]
            # Mean pool over non-padded tokens
            valid = mask.astype(bool)
            if valid.any():
                vec = emb[valid].mean(axis=0)
            else:
                vec = np.zeros(emb.shape[1], dtype=np.float32)
            
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            pooled.append(vec.astype(np.float32))
        
        return pooled
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, return 384-dim float32 array"""
        return self.embed([text])[0]


class HashEmbedder:
    """
    Deterministic hash-based embedder for tests.
    No model required, completely offline, deterministic.
    """
    
    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed
    
    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Deterministic pseudo-random embeddings based on text hash"""
        results = []
        for text in texts:
            # Use hash of text as seed for deterministic output
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16) ^ self.seed
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results
    
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(
    use_hash: bool = False, 
    model_dir: Optional[str] = None,
    **kwargs
):
    """Factory function to get appropriate embedder"""
    if use_hash:
        return HashEmbedder(**kwargs)
    return OnnxEmbedder(model_dir=model_dir)