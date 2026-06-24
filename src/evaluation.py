"""
evaluation.py
-------------
Evaluation metrics for LPS:
  - ROUGE-1 / ROUGE-L
  - BERTScore-F1
  - Topic Coherence (TC)
  - Persona Fidelity (PF)  [in generation.py as LLM-as-judge]
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


class LPSEvaluator:
    """
    Computes all LPS evaluation metrics.

    Parameters
    ----------
    rouge_types : list of str
        ROUGE variants to compute.
    bertscore_model : str
        Model for BERTScore computation.
    """

    def __init__(
        self,
        rouge_types: List[str] = ("rouge1", "rougeL"),
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    ):
        self.rouge_types = list(rouge_types)
        self.bertscore_model = bertscore_model
        self._rouge = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=True
        )

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------
    def rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE-1 and ROUGE-L F1 scores.

        Returns dict with keys 'rouge1', 'rougeL'.
        """
        scores = {t: [] for t in self.rouge_types}
        for pred, ref in zip(predictions, references):
            result = self._rouge.score(ref, pred)
            for t in self.rouge_types:
                scores[t].append(result[t].fmeasure)
        return {t: float(np.mean(v)) for t, v in scores.items()}

    # ------------------------------------------------------------------
    # BERTScore
    # ------------------------------------------------------------------
    def bertscore(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Compute BERTScore-F1 (mean across samples).
        """
        _, _, F1 = bert_score_fn(
            predictions,
            references,
            model_type=self.bertscore_model,
            verbose=False,
        )
        return float(F1.mean().item())

    # ------------------------------------------------------------------
    # Topic Coherence
    # ------------------------------------------------------------------
    @staticmethod
    def topic_coherence(
        response_embeddings: np.ndarray,
        anchor_embeddings: np.ndarray,
    ) -> float:
        """
        Average cosine similarity between generated response embedding
        and the most relevant cognitive anchor centroid.

        Parameters
        ----------
        response_embeddings : np.ndarray, shape (N, d)
        anchor_embeddings : np.ndarray, shape (N, d)
            The centroid of the top-matched anchor for each response.

        Returns
        -------
        float — mean cosine similarity
        """
        # Both inputs assumed L2-normalized
        sims = (response_embeddings * anchor_embeddings).sum(axis=1)
        return float(np.mean(sims))

    # ------------------------------------------------------------------
    # Full evaluation suite
    # ------------------------------------------------------------------
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        pred_embeddings: np.ndarray,
        anchor_embeddings: np.ndarray,
        pf_scores: List[float],
    ) -> Dict[str, float]:
        """
        Run the full evaluation suite and return all metrics.

        Parameters
        ----------
        predictions : list of generated responses A*
        references : list of ground-truth user responses
        pred_embeddings : shape (N, d) — embeddings of predictions
        anchor_embeddings : shape (N, d) — top anchor centroids
        pf_scores : list of Persona Fidelity scores from LLM-as-judge

        Returns
        -------
        dict with keys: rouge1, rougeL, bertscore_f1, topic_coherence, persona_fidelity
        """
        rouge_scores = self.rouge(predictions, references)
        bs_f1 = self.bertscore(predictions, references)
        tc = self.topic_coherence(pred_embeddings, anchor_embeddings)
        pf = float(np.mean(pf_scores)) if pf_scores else 0.0

        return {
            "rouge1":            round(rouge_scores["rouge1"] * 100, 2),
            "rougeL":            round(rouge_scores["rougeL"] * 100, 2),
            "bertscore_f1":      round(bs_f1, 4),
            "topic_coherence":   round(tc, 4),
            "persona_fidelity":  round(pf, 2),
        }
