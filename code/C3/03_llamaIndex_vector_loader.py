from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_index(persist_dir: str):
    path = Path(persist_dir)
    if not path.exists():
        raise FileNotFoundError(f"Persist directory does not exist: {path}")

    # 关键：显式指定本地 embedding，避免默认走 OpenAI
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    storage_context = StorageContext.from_defaults(persist_dir=str(path))
    return load_index_from_storage(storage_context)


def similarity_search(persist_dir: str, query: str, top_k: int = 3):
    index = load_index(persist_dir)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        results = similarity_search(args.persist_dir, args.query, args.top_k)
        for i, item in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Score: {item.score}")
            print(item.node.get_content())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()