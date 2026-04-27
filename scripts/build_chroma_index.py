"""Build the ChromaDB index from the training set.

One-shot script. Loads ``data/processed/train.csv``, deduplicates via
``VectorStore.deduplicate`` (cosine distance < 0.05 = near-dup), then
``VectorStore.build_index`` on what's left. Persists to ``./chroma_db/``.

Run from project root:

    .venv/bin/python -m scripts.build_chroma_index

Most of the runtime (~2-3 min on a laptop CPU) is the one-shot batch
embedding pass over ~11k texts via all-MiniLM-L6-v2.
"""
import os

# default to offline mode: the model was downloaded once during STEP 5
# and is cached under ~/.cache/huggingface. HuggingFace's library
# otherwise does a HEAD request on every load to revalidate ETags,
# which fails on networks with self-signed corporate certs.
# A user can override this by exporting HF_HUB_OFFLINE=0 before running.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path

import pandas as pd

from src.vector_store import VectorStore

TRAIN_CSV = Path("data/processed/train.csv")


def main():
    df = pd.read_csv(TRAIN_CSV)
    print(f"loaded {len(df)} training postings from {TRAIN_CSV}")

    texts = df["job_text"].astype(str).tolist()
    labels = df["fraudulent"].astype(int).tolist()
    ids = df["job_id"].astype(int).tolist()

    vs = VectorStore()  # writes to ./chroma_db

    print("deduplicating (cosine distance < 0.05)...")
    kept_texts, kept_labels, kept_ids = vs.deduplicate(texts, labels, ids=ids)
    dropped = len(texts) - len(kept_texts)
    print(f"  kept {len(kept_texts)} / dropped {dropped} near-duplicates")

    fraud_kept = sum(kept_labels)
    print(f"  fraud rate after dedup: {fraud_kept / len(kept_labels):.2%}")

    print(f"indexing into ./chroma_db (collection {vs.collection.name})...")
    vs.build_index(kept_texts, kept_labels, kept_ids)
    print(f"  collection now has {vs.collection.count()} documents")
    print("done.")


if __name__ == "__main__":
    main()
