# Study sheet — `src/vector_store.py`

This file is the only place ChromaDB lives in the project. It does
three things, **none of them RAG**:

1. Powers the kNN risk feature (`neighbor_fraud_rate`).
2. Backs the Streamlit "similar past postings" sidebar (`query`).
3. Runs the training-set deduplication before fine-tuning
   (`deduplicate`).

## What each piece does

### Top-of-file constants

- `DEFAULT_PATH = "./chroma_db"` — PersistentClient writes embeddings
  to this folder on disk so we don't re-embed 11k postings every time
  the kernel restarts. The folder is gitignored.
- `COLLECTION_NAME = "trustlens"` — chroma's term for what is
  basically a table.
- `MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"` — small
  (80 MB), fast, CPU-friendly, produces 384-dim embeddings.

### `__init__(self, path)`

Opens a `PersistentClient` rooted at `path`, builds a
`SentenceTransformerEmbeddingFunction` with our model name, and gets
(or creates) the trustlens collection. The one non-default detail is
`metadata={"hnsw:space": "cosine"}` — chroma's default index distance
is L2; we override to cosine because (a) all-MiniLM was trained on a
cosine-similarity objective, (b) the dedup threshold of 0.05 is
calibrated against cosine and would mean nothing in L2.

### `build_index(texts, labels, ids, batch_size=200)`

Inserts every training posting in batches of 200. Uses `upsert` (not
`add`) so a re-run on an existing collection updates rows in place
instead of raising. Each row carries its label in metadata so we can
read it back during `query`.

### `query(text, k=5)`

Calls `collection.query`, then reshapes chroma's columns-of-lists
return shape into the row-of-dicts shape the rest of the project
wants: `{id, text, label, distance}`. The Streamlit sidebar renders
this directly; `neighbor_fraud_rate` consumes it.

### `neighbor_fraud_rate(text, k=5)`

Asks `query` for the k nearest neighbors, then returns the mean of
their labels (0 = real, 1 = fraud). Output is a float in [0, 1].

This is the **only** ChromaDB output that ever flows into an LLM
prompt — and it flows in as a number, not as text. That single fact
is what keeps this design out of RAG territory.

### `deduplicate(texts, labels, ids=None, threshold=0.05)`

- Embed every text in one batch via `self.embedder`.
- L2-normalise each vector so cosine similarity reduces to a plain
  dot product.
- Walk through the list keeping each row only if its cosine distance
  to every already-kept row is `>= threshold`.

If `ids` is passed, the function returns `(kept_texts, kept_labels,
kept_ids)` so the caller can hand the result straight to
`build_index`. Otherwise it returns `(kept_texts, kept_labels)`.

`threshold=0.05` means "if two postings differ by less than 5% of
their cosine angle, drop the second one." That catches reposted scams
with a word or two changed but does not collide with genuinely
different postings that happen to share boilerplate.

## Sample viva Q&A

**Q: Why is this not RAG?**
A: RAG = retrieve relevant documents and concatenate them into the
LLM's prompt as context. We never do that. This file returns one of
three things: a single float (`neighbor_fraud_rate`), a list of dicts
that the Streamlit sidebar renders for the user (`query`, used at the
UI layer only), or a filtered training set (`deduplicate`). None of
those is prompt-augmentation. The LLM sees the raw posting plus a few
numeric features — never retrieved text from past postings.

**Q: Why all-MiniLM-L6-v2 specifically?**
A: Small (80 MB on disk), fast on CPU (~75 sentences/second on a
laptop), 384-dim — small enough that ChromaDB's HNSW index stays fast
at our 11k-row scale. Larger embedders like all-mpnet-base-v2 score
a few points higher on benchmarks but are roughly 4× slower and we
have no GPU at inference time. The accuracy gain isn't worth the
latency.

**Q: Why cosine distance and not L2?**
A: All-MiniLM was trained with a cosine-similarity objective, so
cosine is the metric the embedding space is calibrated for. Two
semantically similar postings will land at near-zero cosine distance
even if their L2 distance varies (transformer embeddings aren't
unit-length by default). Our dedup threshold of 0.05 is also
calibrated against cosine — it would be meaningless in L2.
