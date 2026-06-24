"""
graph_rag.py
------------
Relational Persona Graph-RAG (P-Graph) — Framework II of LPS.

Implements:
  1. Triplet extraction from each interaction: (s, p, o, κ)
  2. Knowledge graph construction with temporal edge-weight decay:

        W(e_uv) = Σ_{t ∈ T_uv}  γ^(t_now - τ_t)

  3. Leiden hierarchical community detection → Persona Sub-graphs
  4. Multi-tier retrieval:
       Tier 1 — Global alignment via community centroid similarity
       Tier 2 — Structural summarization via d-hop traversal
       Tier 3 — Exemplar extraction of high-weight triplets
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import anthropic


@dataclass
class PersonaTriplet:
    subject: str
    predicate: str
    obj: str
    kappa: str          # meta-contextual tone (e.g. 'asserting', 'questioning')
    timestamp: float
    embedding: Optional[np.ndarray] = None


@dataclass
class PersonaCommunity:
    community_id: int
    nodes: List[str]
    centroid: np.ndarray
    summary: Optional[str] = None


class PersonaGraphRAG:
    """
    Relational Persona Graph-RAG framework.

    Parameters
    ----------
    gamma_decay : float
        Geometric decay factor γ ∈ (0,1) for edge weights.
    leiden_resolution : float
        Resolution parameter for Leiden community detection.
    hop_depth : int
        d-hop depth for structural summarization traversal.
    """

    def __init__(
        self,
        gamma_decay: float = 0.95,
        leiden_resolution: float = 1.2,
        hop_depth: int = 2,
        anthropic_client: Optional[anthropic.Anthropic] = None,
    ):
        self.gamma_decay = gamma_decay
        self.leiden_resolution = leiden_resolution
        self.hop_depth = hop_depth
        self.client = anthropic_client or anthropic.Anthropic()
        self.graph_ = nx.DiGraph()
        self.triplets_: List[PersonaTriplet] = []
        self.communities_: List[PersonaCommunity] = []

    # ------------------------------------------------------------------
    # Step 1: Extract triplets from a single interaction via LLM
    # ------------------------------------------------------------------
    def extract_triplets(self, user_turn: str, timestamp: float) -> List[PersonaTriplet]:
        """
        Use Claude to extract (subject, predicate, object, tone) triplets
        from a user turn.
        """
        prompt = f"""Extract structured knowledge triplets from this user message.
For each triplet output JSON with keys: subject, predicate, object, tone.
Tone must be one of: asserting, questioning, disagreeing, exploring, confirming.
Output a JSON array only, no explanation.

User message: \"{user_turn}\""""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        import json
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []

        triplets = []
        for item in items:
            triplets.append(PersonaTriplet(
                subject=item.get("subject", ""),
                predicate=item.get("predicate", ""),
                obj=item.get("object", ""),
                kappa=item.get("tone", "asserting"),
                timestamp=timestamp,
            ))
        return triplets

    # ------------------------------------------------------------------
    # Step 2: Build / update the knowledge graph
    # ------------------------------------------------------------------
    def build_graph(
        self,
        triplets: List[PersonaTriplet],
        t_now: Optional[float] = None,
    ) -> "PersonaGraphRAG":
        """
        Construct the knowledge graph from a list of triplets with
        temporally decayed edge weights.
        """
        if t_now is None:
            t_now = max(t.timestamp for t in triplets)

        self.triplets_ = triplets
        self.graph_.clear()

        # Group triplets by (subject, object) pair
        edge_map: Dict[Tuple[str, str], List[PersonaTriplet]] = {}
        for t in triplets:
            key = (t.subject, t.obj)
            edge_map.setdefault(key, []).append(t)

        for (subj, obj), trip_list in edge_map.items():
            # W(e_uv) = Σ γ^(t_now - τ_t)
            weight = sum(
                self.gamma_decay ** (t_now - t.timestamp)
                for t in trip_list
            )
            predicates = [t.predicate for t in trip_list]
            kappas = [t.kappa for t in trip_list]

            self.graph_.add_edge(
                subj, obj,
                weight=weight,
                predicates=predicates,
                kappas=kappas,
                triplets=trip_list,
            )

        return self

    # ------------------------------------------------------------------
    # Step 3: Leiden community detection
    # ------------------------------------------------------------------
    def detect_communities(
        self, embeddings_map: Dict[str, np.ndarray]
    ) -> "PersonaGraphRAG":
        """
        Run Leiden community detection and compute community centroids.

        Parameters
        ----------
        embeddings_map : dict mapping node name -> embedding vector
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Install igraph and leidenalg: pip install igraph leidenalg")

        nodes = list(self.graph_.nodes())
        if not nodes:
            return self

        # Build igraph from networkx
        ig_graph = ig.Graph.from_networkx(self.graph_.to_undirected())
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.leiden_resolution,
        )

        self.communities_ = []
        for cid, community in enumerate(partition):
            community_nodes = [nodes[i] for i in community]
            # Compute centroid from node embeddings
            node_embs = [
                embeddings_map[n] for n in community_nodes
                if n in embeddings_map
            ]
            if node_embs:
                centroid = np.mean(node_embs, axis=0)
                centroid /= (np.linalg.norm(centroid) + 1e-9)
            else:
                centroid = np.zeros(768)

            self.communities_.append(PersonaCommunity(
                community_id=cid,
                nodes=community_nodes,
                centroid=centroid,
            ))

        return self

    # ------------------------------------------------------------------
    # Step 4: Multi-tier retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k_communities: int = 2,
        top_k_triplets: int = 10,
    ) -> Dict:
        """
        Three-tier relational retrieval for a query.

        Returns
        -------
        dict with keys:
            'community_prior'  — top community centroids (Tier 1)
            'subgraph_summary' — d-hop relational summary (Tier 2)
            'exemplar_triplets'— high-weight triplets (Tier 3)
        """
        # Tier 1: Global community alignment
        if self.communities_:
            centroids = np.stack([c.centroid for c in self.communities_])
            sims = centroids @ query_embedding
            top_idx = np.argsort(sims)[::-1][:top_k_communities]
            top_communities = [self.communities_[i] for i in top_idx]
        else:
            top_communities = []

        # Tier 2: d-hop subgraph traversal around top community nodes
        subgraph_nodes = set()
        for community in top_communities:
            for node in community.nodes[:5]:  # Seed from top community nodes
                if node in self.graph_:
                    neighbors = nx.single_source_shortest_path_length(
                        self.graph_, node, cutoff=self.hop_depth
                    )
                    subgraph_nodes.update(neighbors.keys())

        subgraph = self.graph_.subgraph(subgraph_nodes)
        subgraph_edges = [
            (u, v, self.graph_[u][v]) for u, v in subgraph.edges()
        ]

        # Tier 3: Extract highest-weight triplets
        all_triplets = []
        for _, _, data in self.graph_.edges(data=True):
            for t in data.get("triplets", []):
                w = self.gamma_decay ** (max(tr.timestamp for tr in self.triplets_) - t.timestamp)
                all_triplets.append((w, t))

        all_triplets.sort(key=lambda x: x[0], reverse=True)
        top_triplets = [t for _, t in all_triplets[:top_k_triplets]]

        return {
            "community_prior": top_communities,
            "subgraph_edges": subgraph_edges,
            "exemplar_triplets": top_triplets,
        }
