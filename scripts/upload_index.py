"""Push chroma_db/ to a free HF *dataset* repo.

Streamlit Community Cloud deploys from GitHub, and chroma.sqlite3 is
173 MB — over GitHub's hard 100 MB per-file limit. Dataset repos are
free and have no such limit, so the index lives there and the app
pulls it on first boot (see src/vector_store.py::_ensure_index).

Usage:
    HF_WRITE_TOKEN=hf_xxx .venv/bin/python scripts/upload_index.py
"""
import os
import sys

from huggingface_hub import HfApi

DATASET_ID = "SAANVEE/trustlens-chroma"


def main():
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        sys.exit("Set HF_WRITE_TOKEN to a WRITE-scoped token.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=DATASET_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    print(f"dataset repo ready: {DATASET_ID}")

    api.upload_folder(
        repo_id=DATASET_ID,
        repo_type="dataset",
        folder_path="chroma_db",
        path_in_repo="",
        commit_message="chroma index for TrustLens demo",
    )
    print(f"uploaded → https://huggingface.co/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
