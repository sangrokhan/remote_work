#!/usr/bin/env python3
"""Download BGE embedding and reranker models to local cache."""

import argparse
import os
import sys
from pathlib import Path


MODELS = {
    "embedder": "BAAI/bge-large-en-v1.5",
    "reranker": "BAAI/bge-reranker-v2-m3",
}

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "models"


def download_model(model_id: str, cache_dir: Path, force: bool = False) -> None:
    from huggingface_hub import snapshot_download

    model_name = model_id.split("/")[-1]
    target = cache_dir / model_name

    if target.exists() and not force:
        print(f"[skip] {model_id} already exists at {target}")
        return

    print(f"[download] {model_id} → {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[done] {model_id}")


def verify_model(model_id: str, cache_dir: Path) -> bool:
    """Quick load check — ensures tokenizer + model weights are loadable."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    model_name = model_id.split("/")[-1]
    target = str(cache_dir / model_name)

    try:
        if "reranker" in model_id.lower():
            CrossEncoder(target)
        else:
            model = SentenceTransformer(target)
            _ = model.encode(["test sentence"], show_progress_bar=False)
        print(f"[ok] {model_id} verified")
        return True
    except Exception as e:
        print(f"[fail] {model_id}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BGE models for SPAR")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to download (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Local model cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if model already exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a quick load check after download",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token, add_to_git_credential=False)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    targets = list(MODELS.keys()) if "all" in args.models else args.models

    failed = []
    for key in targets:
        model_id = MODELS[key]
        try:
            download_model(model_id, args.cache_dir, force=args.force)
            if args.verify:
                if not verify_model(model_id, args.cache_dir):
                    failed.append(model_id)
        except Exception as e:
            print(f"[error] {model_id}: {e}", file=sys.stderr)
            failed.append(model_id)

    if failed:
        print(f"\nFailed: {failed}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll models in: {args.cache_dir}")
    print("Set ENCODER_MODEL env var to use:")
    embedder_path = args.cache_dir / MODELS["embedder"].split("/")[-1]
    print(f"  export ENCODER_MODEL={embedder_path}")


if __name__ == "__main__":
    main()
