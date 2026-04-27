"""ChromaDB wrapper for kNN risk features, UI similarity, and dedup.

This is intentionally NOT RAG. ChromaDB never feeds retrieved text
back into the LLM prompt. Its three jobs:

1. ``neighbor_fraud_rate(text)`` returns a single float in [0, 1] —
   the fraction of the K nearest training postings labelled fraud.
   That number goes into the LLM prompt as a numeric feature.
2. ``query(text, k)`` returns a list of ``{id, text, label, distance}``
   dicts. Used by the Streamlit UI to show similar past postings.
3. ``deduplicate(texts, labels, threshold)`` drops near-duplicates
   before fine-tuning so the model never trains on copy-paste scams.
"""
import chromadb
from chromadb.utils import embedding_functions

DEFAULT_PATH = "./chroma_db"
COLLECTION_NAME = "trustlens"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, path: str = DEFAULT_PATH):
        # PersistentClient writes to disk so we don't re-embed the whole
        # training set every time the kernel restarts.
        self.client = chromadb.PersistentClient(path=str(path))
        # chroma's helper downloads all-MiniLM-L6-v2 the first time and
        # caches it under ~/.cache/huggingface.
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
        )
        # default distance is L2; we override to cosine because
        # all-MiniLM was trained on a cosine-similarity objective and
        # our dedup threshold of 0.05 only makes sense in cosine space.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, texts, labels, ids, batch_size: int = 200):
        """Embed and store every (text, label, id) triple.

        Uses upsert (not add) so re-running on existing IDs updates
        them in place instead of raising.
        """
        assert len(texts) == len(labels) == len(ids), "length mismatch"
        texts = list(texts)
        labels = list(labels)
        ids = list(ids)
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            self.collection.upsert(
                ids=[str(i) for i in ids[start:stop]],
                documents=texts[start:stop],
                metadatas=[{"label": int(l)} for l in labels[start:stop]],
            )

    def query(self, text: str, k: int = 5):
        """Return the k nearest postings as a list of dicts.

        Chroma returns columns-of-lists; we reshape to a row-of-dicts
        list because that's what the Streamlit sidebar and
        neighbor_fraud_rate want.
        """
        result = self.collection.query(query_texts=[text], n_results=k)
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        out = []
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({
                "id": i,
                "text": doc,
                "label": int(meta["label"]),
                "distance": float(dist),
            })
        return out

    def neighbor_fraud_rate(self, text: str, k: int = 5) -> float:
        """Fraction of the k nearest training postings labelled fraud.

        This is the only ChromaDB output that flows into the LLM —
        and it flows in as a number, not text. That single fact is
        what keeps this design out of RAG territory.
        """
        neighbors = self.query(text, k=k)
        if not neighbors:
            return 0.0
        return sum(n["label"] for n in neighbors) / len(neighbors)

    def deduplicate(self, texts, labels, ids=None, threshold: float = 0.05):
        """Drop near-duplicate postings before fine-tuning.

        Strategy:
        1. Embed every text in one batch.
        2. L2-normalise so cosine similarity equals a dot product.
        3. Walk the list, keeping a row only if its cosine distance
           to every already-kept row is >= threshold.

        Returns ``(kept_texts, kept_labels)`` if ``ids`` is None,
        otherwise ``(kept_texts, kept_labels, kept_ids)`` so the
        caller can pass them straight to ``build_index``.
        """
        import numpy as np
        texts = list(texts)
        labels = list(labels)
        embs = np.asarray(self.embedder(texts), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms == 0, 1, norms)

        kept = []
        for i, e in enumerate(embs):
            if kept:
                sims = embs[kept] @ e
                if float((1.0 - sims).min()) < threshold:
                    continue
            kept.append(i)
        kept_texts = [texts[i] for i in kept]
        kept_labels = [labels[i] for i in kept]
        if ids is None:
            return kept_texts, kept_labels
        ids = list(ids)
        return kept_texts, kept_labels, [ids[i] for i in kept]
