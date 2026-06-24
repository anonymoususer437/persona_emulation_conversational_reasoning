"""
clustering.py
-------------
Evolutionary Latent Clustering (ELC) — Framework I of LPS.

Implements:
  1. HDBSCAN clustering over user-prioritized embeddings → topical clusters {S_j}
  2. Time-decayed centroid (Cognitive Anchor) per cluster:

        μ_j = Σ_{t ∈ S_j} v_t · exp(-λ(t_now - τ_t))
              ─────────────────────────────────────────
              Σ_{t ∈ S_j} exp(-λ(t_now - τ_t))

  3. Query-to-anchor matching via cosine similarity for Tier 1 retrieval.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import hdbscan


@dataclass
class CognitiveAnchor:
    """Represents a single time-decayed cluster centroid (Cognitive Anchor)."""
    cluster_id: int
    centroid: np.ndarray          # μ_j — time-decayed centroid vector
    member_indices: List[int]     # Indices into the interaction list
    member_weights: List[float]   # Temporal decay weights for each member
    label: Optional[str] = None   # Optional human-readable topic label


class ELCFramework:
    """
    Evolutionary Latent Clustering framework.

    Parameters
    ----------
    lambda_decay : float
        Exponential decay constant λ. Default 0.01 ≈ half-life of 69 days.
    min_cluster_size : int
        HDBSCAN min_cluster_size.
    min_samples : int
        HDBSCAN min_samples.
    """

    def __init__(
        self,
        lambda_decay: float = 0.01,
        min_cluster_size: int = 5,
        min_samples: int = 3,
    ):
        self.lambda_decay = lambda_decay
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.anchors_: List[CognitiveAnchor] = []
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1: Cluster
    # ------------------------------------------------------------------
    def fit(
        self,
        embeddings: np.ndarray,
        timestamps: List[float],
        t_now: Optional[float] = None,
    ) -> "ELCFramework":
        """
        Fit the ELC framework to a user's interaction history.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, d)
            User-prioritized embeddings, one per interaction.
        timestamps : list of float
            Unix timestamps for each interaction (τ_t).
        t_now : float, optional
            Reference time for decay. Defaults to max(timestamps).
        """
        if t_now is None:
            t_now = max(timestamps)

        timestamps = np.array(timestamps, dtype=float)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            prediction_data=True,
        )
        self.labels_ = clusterer.fit_predict(embeddings)

        # Build Cognitive Anchors
        self.anchors_ = []
        unique_labels = set(self.labels_)
        unique_labels.discard(-1)  # Remove noise label

        for cid in sorted(unique_labels):
            member_idx = np.where(self.labels_ == cid)[0].tolist()
            member_ts = timestamps[member_idx]

            # Temporal decay weights: exp(-λ · Δt)
            delta_t = t_now - member_ts
            weights = np.exp(-self.lambda_decay * delta_t)
            weights /= weights.sum()  # Normalize

            # Time-decayed centroid μ_j
            member_embs = embeddings[member_idx]  # (|S_j|, d)
            centroid = (member_embs * weights[:, None]).sum(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-9)

            self.anchors_.append(
                CognitiveAnchor(
                    cluster_id=cid,
                    centroid=centroid,
                    member_indices=member_idx,
                    member_weights=weights.tolist(),
                )
            )

        return self

    # ------------------------------------------------------------------
    # Step 2: Retrieve top-k anchors for a query (Tier 1)
    # ------------------------------------------------------------------
    def retrieve_anchors(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[CognitiveAnchor]:
        """
        Return the top-k Cognitive Anchors most similar to the query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (d,)
        top_k : int

        Returns
        -------
        List of CognitiveAnchor sorted by descending cosine similarity.
        """
        if not self.anchors_:
            raise RuntimeError("Call fit() before retrieve_anchors().")

        centroids = np.stack([a.centroid for a in self.anchors_])  # (K, d)
        sims = centroids @ query_embedding                          # (K,)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.anchors_[i] for i in top_idx]

    # ------------------------------------------------------------------
    # Step 3: Retrieve exemplar interactions from top anchors
    # ------------------------------------------------------------------
    def retrieve_exemplars(
        self,
        query_embedding: np.ndarray,
        interactions: List[Dict],
        top_k_anchors: int = 3,
        top_k_exemplars: int = 5,
    ) -> List[Dict]:
        """
        Retrieve the highest-weighted exemplar interactions from the top-k anchors.

        Parameters
        ----------
        interactions : list of dicts with keys 'user', 'model', 'timestamp', 'embedding'
        """
        top_anchors = self.retrieve_anchors(query_embedding, top_k=top_k_anchors)
        candidates = []
        for anchor in top_anchors:
            for idx, weight in zip(anchor.member_indices, anchor.member_weights):
                candidates.append((weight, interactions[idx]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in candidates[:top_k_exemplars]]
