# src/claim_clustering.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

from src.claims_extraction import Claim


@dataclass
class ClaimCluster:
    """
    Repräsentiert einen Cluster von inhaltlich ähnlichen Claims.
    - cluster_id: numerische Cluster-ID (von K-Means)
    - claim_indices: Indizes der Claims in der ursprünglichen Claim-Liste
    - claims: die zugehörigen Claim-Objekte
    - centroid: Schwerpunkt des Clusters im Embedding-Raum
    - claim_topics: thematische Tags (wird in claim_topics.py gesetzt)
    """
    cluster_id: int
    claim_indices: List[int]
    claims: List[Claim]
    centroid: List[float]
    claim_topics: List[str]


class ClaimEmbedder:
    """
    Erzeugt Satz-Embeddings für Claims mittels eines Transformer-Encoders.

    Standardmäßig wird ein generisches Sentence-Embedding-Modell verwendet.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Gibt ein numpy-Array der Form (n_sätze, embedding_dim) zurück.
        Embeddings werden über Mean-Pooling der Token-Embeddings erzeugt.
        """
        if not sentences:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            # last_hidden_state: (batch, seq_len, hidden_dim)
            last_hidden = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked = last_hidden * attention_mask                               # Maskierung der Padding-Tokens

            sum_embeddings = masked.sum(dim=1)                                  # (batch, hidden_dim)
            lengths = attention_mask.sum(dim=1)                                 # (batch, 1)
            mean_embeddings = sum_embeddings / torch.clamp(lengths, min=1e-9)   # Mean-Pooling

            all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class ClaimClusterer:
    """
    Führt Clustering von Claims im Embedding-Raum durch.

    Typischerweise rufst du:
        embedder = ClaimEmbedder(...)
        clusterer = ClaimClusterer(embedder, n_clusters=10)
        clusters = clusterer.cluster_claims(claims)
    """

    def __init__(
        self,
        embedder: ClaimEmbedder,
        n_clusters: int = 10,
        random_state: int = 42,
    ) -> None:
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster_claims(self, claims: List[Claim]) -> List[ClaimCluster]:
        """
        Führt K-Means auf den Claim-Embeddings aus und gruppiert Claims zu Clustern.
        """
        if not claims:
            return []

        sentences = [c.sentence for c in claims]
        embeddings = self.embedder.encode(sentences)  # shape: (n_claims, dim)

        # Sicherheitsfall: weniger Claims als Clusterzahl
        n_clusters = min(self.n_clusters, embeddings.shape[0])

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(embeddings)

        tmp_clusters: Dict[int, Dict[str, Any]] = {}
        for idx, (label, emb) in enumerate(zip(labels, embeddings)):
            if label not in tmp_clusters:
                tmp_clusters[label] = {
                    "indices": [],
                    "claims": [],
                    "embeddings": [],
                }
            tmp_clusters[label]["indices"].append(idx)
            tmp_clusters[label]["claims"].append(claims[idx])
            tmp_clusters[label]["embeddings"].append(emb)

        clusters: List[ClaimCluster] = []
        for cid, data in tmp_clusters.items():
            centroid = np.mean(np.vstack(data["embeddings"]), axis=0)
            clusters.append(
                ClaimCluster(
                    cluster_id=int(cid),
                    claim_indices=data["indices"],
                    claims=data["claims"],
                    centroid=centroid.tolist(),
                    claim_topics=[],  # wird in claim_topics.py ergänzt
                )
            )

        # Konsistente Sortierung nach cluster_id
        clusters.sort(key=lambda c: c.cluster_id)
        return clusters
