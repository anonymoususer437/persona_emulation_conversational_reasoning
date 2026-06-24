"""
retrieval.py
------------
Two-Stage Retrieval pipeline shared by both LPS frameworks.

Tier 1 — Semantic Topic Match:
    Match query embedding against Cognitive Anchors (ELC) or
    community centroids (Graph-RAG) via cosine similarity.

Tier 2 — Meta-Contextual Tone Match (κ):
    Within retrieved domain, filter exemplars by stylistic tone
    that best matches the implied tone of the query.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


TONE_LABELS = ["asserting", "questioning", "disagreeing", "exploring", "confirming"]


class TwoStageRetriever:
    """
    Performs two-stage retrieval over exemplar interactions.

    Parameters
    ----------
    encoder : SentenceTransformer
        Shared sentence encoder for tone embedding.
    top_k_exemplars : int
        Final number of exemplars to return after both tiers.
    """

    def __init__(
        self,
        encoder: Optional[SentenceTransformer] = None,
        top_k_exemplars: int = 5,
    ):
        self.encoder = encoder or SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.top_k_exemplars = top_k_exemplars

        # Pre-compute tone anchor embeddings for Tier 2
        self._tone_anchors = self.encoder.encode(
            TONE_LABELS, normalize_embeddings=True
        )

    # ------------------------------------------------------------------
    # Tier 2: Detect the dominant tone of a query
    # ------------------------------------------------------------------
    def _infer_query_tone(self, query_embedding: np.ndarray) -> str:
        """
        Infer the meta-contextual tone κ of the query by cosine similarity
        against tone label embeddings.
        """
        sims = self._tone_anchors @ query_embedding
        return TONE_LABELS[int(np.argmax(sims))]

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        tier1_exemplars: List[Dict],
    ) -> Dict:
        """
        Run the two-stage retrieval pipeline.

        Parameters
        ----------
        query : str
            Raw query string (used for tone inference).
        query_embedding : np.ndarray, shape (d,)
            Encoded query vector.
        tier1_exemplars : list of dicts
            Candidate exemplars from Tier 1 (anchor-matched).
            Each dict must contain: 'user', 'model', 'kappa', 'timestamp'.

        Returns
        -------
        dict with keys:
            'query_tone'    — inferred tone κ of the query
            'logic_stream'  — exemplars matching topic (Tier 1 result)
            'style_stream'  — exemplars matching tone κ (Tier 2 filtered)
        """
        # Tier 1 result: exemplars already matched by topic
        logic_stream = tier1_exemplars[:self.top_k_exemplars]

        # Tier 2: infer query tone and filter by matching κ
        query_tone = self._infer_query_tone(query_embedding)

        style_stream = [
            ex for ex in tier1_exemplars
            if ex.get("kappa", "") == query_tone
        ]

        # Fall back to full Tier 1 pool if no tone matches
        if not style_stream:
            style_stream = logic_stream

        style_stream = style_stream[:self.top_k_exemplars]

        return {
            "query_tone": query_tone,
            "logic_stream": logic_stream,
            "style_stream": style_stream,
        }

    # ------------------------------------------------------------------
    # Format context for LLM prompt
    # ------------------------------------------------------------------
    @staticmethod
    def format_context(retrieval_result: Dict) -> str:
        """
        Format the two-stream retrieval result into a structured prompt prefix.
        """
        lines = ["=== LOGIC STREAM (User's reasoning and knowledge) ==="]
        for i, ex in enumerate(retrieval_result["logic_stream"], 1):
            lines.append(f"[{i}] User: {ex.get('user', '')}")
            lines.append(f"    (Tone: {ex.get('kappa', 'N/A')})")

        lines.append("\n=== STYLE STREAM (User's characteristic tone and style) ===")
        for i, ex in enumerate(retrieval_result["style_stream"], 1):
            lines.append(f"[{i}] User: {ex.get('user', '')}")

        lines.append(f"\nInferred query tone: {retrieval_result['query_tone']}")
        return "\n".join(lines)
