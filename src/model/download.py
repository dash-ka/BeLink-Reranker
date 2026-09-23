import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

HF_REPO_ID = "Dash00/belink-reranker"
CACHE_DIR = Path(os.environ.get("BELINK_RERANKER_CACHE", Path.home() / ".cache" / "belink_reranker"))


def get_model_path(force_download: bool = False, token: Optional[str] = None) -> str:
    """
    Downloads (if needed) and returns the local path to the model weights.

    token: Hugging Face access token. Only required if the model repo is
           private. If the repo is public, leave this as None.
    """
    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        cache_dir=CACHE_DIR,
        force_download=force_download,
        token=token,
    )
    return local_path
