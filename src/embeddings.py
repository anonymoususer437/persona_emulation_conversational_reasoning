"""
embeddings.py
-------------
User-Prioritized Embedding (Stage 1 of LPS).

Implements:
    v_t = α · enc(U_t) + (1 - α) · enc(M_t)

where α ∈ [0.5, 1.0] is the Agency Coefficient that amplifies the user's
own turns relative to model-generated verbosity, directly countering
Model Verbosity Bias.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class UserPrioritizedEmbedder:
    """
    Computes agency-weighted embeddings for each (user_turn, model_turn) pair.

    Parameters
    ----------
    model_name : str
        Sentence encoder model name (default: text-embedding-3-large via ST wrapper).
    alpha : float
        Agency Coefficient in [0.5, 1.0]. Higher values prioritize user turns.
    batch_size : int
        Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        alpha: float = 0.75,
        batch_size: int = 64,
    ):
        assert 0.5 <= alpha <= 1.0, "Alpha must be in [0.5, 1.0]"
        self.alpha = alpha
        self.batch_size = batch_size
        self.encoder = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into embeddings."""
        return self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_interactions(
        self,
        interactions: List[Tuple[str, str]],
    ) -> np.ndarray:
        """
        Compute User-Prioritized Embeddings for a list of (user_turn, model_turn) pairs.

        Parameters
        ----------
        interactions : list of (user_turn, model_turn) tuples

        Returns
        -------
        np.ndarray of shape (N, d) — one embedding per interaction
        """
        user_texts = [u for u, _ in interactions]
        model_texts = [m for _, m in interactions]

        user_embs = self.encode(user_texts)    # (N, d)
        model_embs = self.encode(model_texts)  # (N, d)

        # v_t = α · enc(U_t) + (1 - α) · enc(M_t)
        weighted = self.alpha * user_embs + (1 - self.alpha) * model_embs

        # Re-normalize after weighting
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return weighted / norms
