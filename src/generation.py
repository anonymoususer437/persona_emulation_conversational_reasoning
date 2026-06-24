"""
generation.py
-------------
Dual-Stream Contextualization and Response Generation (Stage 4 of LPS).

Grounding the LLM generator in both the Logic Stream and Style Stream
retrieved by the Two-Stage Retrieval pipeline to produce a
persona-consistent emulated response A*.
"""

from __future__ import annotations
import anthropic
from typing import Dict, Optional


SYSTEM_PROMPT = """You are an AI system performing Zero-Shot Persona Emulation.
Your task is to predict how a SPECIFIC USER would respond to the given query,
based on their past conversation history provided as context.

You are given two streams of context from the user's history:
1. LOGIC STREAM: Examples showing the user's domain knowledge, reasoning patterns,
   and factual stances on related topics.
2. STYLE STREAM: Examples showing the user's characteristic tone, rhetorical style,
   and level of technical depth.

Generate a response that:
- Reflects the user's OWN reasoning style and intellectual perspective
- Uses the same level of technical depth and vocabulary as the user typically employs
- Maintains the user's characteristic tone (as indicated by the inferred query tone)
- Does NOT sound like a generic AI assistant — sound like THIS specific user

Important: Base your response on the provided context. Do not invent facts or stances
not supported by the user's history."""


class DualStreamGenerator:
    """
    Generates persona-consistent responses using dual-stream context.

    Parameters
    ----------
    model : str
        Anthropic model to use for generation.
    max_tokens : int
        Maximum tokens for generated response.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1000,
        client: Optional[anthropic.Anthropic] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def generate(
        self,
        query: str,
        retrieval_result: Dict,
        context_str: Optional[str] = None,
    ) -> str:
        """
        Generate a persona-consistent response A* for the given query.

        Parameters
        ----------
        query : str
            The novel query Q to emulate a user response for.
        retrieval_result : dict
            Output from TwoStageRetriever.retrieve(), containing
            logic_stream, style_stream, and query_tone.
        context_str : str, optional
            Pre-formatted context string. If None, will be built from
            retrieval_result automatically.

        Returns
        -------
        str — the emulated response A*
        """
        if context_str is None:
            from retrieval import TwoStageRetriever
            context_str = TwoStageRetriever.format_context(retrieval_result)

        user_message = f"""{context_str}

=== TARGET QUERY ===
{query}

=== YOUR TASK ===
Based on the user's history above, generate how THIS specific user would respond
to the target query. Match their reasoning style, tone ({retrieval_result.get('query_tone', 'neutral')}),
and intellectual depth."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def generate_with_pf_judge(
        self,
        query: str,
        generated_response: str,
        ground_truth_response: str,
        n_runs: int = 3,
    ) -> float:
        """
        Compute Persona Fidelity (PF) score using LLM-as-judge (1-5 scale).
        Averages across n_runs independent judge queries.

        Parameters
        ----------
        query : str
        generated_response : str — the emulated response A*
        ground_truth_response : str — the user's actual response
        n_runs : int — number of independent judge calls

        Returns
        -------
        float — mean PF score in [1, 5]
        """
        judge_prompt = f"""You are evaluating how well an AI system emulates a specific user's
persona when responding to a query.

Query: {query}

Generated response (AI emulation):
{generated_response}

Ground truth response (actual user):
{ground_truth_response}

Rate the Persona Fidelity of the generated response on a scale of 1-5:
1 = Generic, sounds like a default AI assistant, no resemblance to user style
2 = Slight resemblance to user style but mostly generic
3 = Moderate match to user's reasoning style and tone
4 = Strong match — captures user's characteristic voice and reasoning
5 = Excellent — nearly indistinguishable from the user's actual response style

Respond with ONLY a single integer (1, 2, 3, 4, or 5)."""

        scores = []
        for _ in range(n_runs):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            try:
                score = int(resp.content[0].text.strip())
                if 1 <= score <= 5:
                    scores.append(score)
            except ValueError:
                pass

        return float(np.mean(scores)) if scores else 3.0


# Make np available
import numpy as np
