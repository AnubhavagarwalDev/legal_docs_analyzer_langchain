import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

# ---------------------------------------------------------
# Persistent cache (JSON on disk)
# ---------------------------------------------------------
_CACHE: Dict[str, Any] = {}
_LOCK = Lock()
_CACHE_FILE = Path(__file__).resolve().parent.parent / "data_internal" / "cache_store.json"
_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_cache_from_disk() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _CACHE.update(data)
    except Exception:
        # Fail-soft: if the file is corrupt, ignore and start fresh
        return


def _save_cache_to_disk() -> None:
    try:
        with _CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(_CACHE, f)
    except Exception:
        # Fail-soft: do not crash the app if disk write fails
        return


# Load once on import
_load_cache_from_disk()


# ---------------------------------------------------------
# Public helpers
# ---------------------------------------------------------
def compute_clause_hash(
    clause: Dict[str, Any],
    prompt_version: str,
    model_name: str = "",
    include_heading: bool = True
) -> str:
    """
    Build a stable hash for a clause so identical text + prompt/model reuse results.
    """
    text = (clause.get("text") or "").strip()
    heading = (clause.get("heading") or "").strip() if include_heading else ""
    key_material = "||".join([
        prompt_version,
        model_name or "",
        heading,
        text
    ])
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def get_cached_result(task: str, cache_key: str) -> Optional[Any]:
    """
    Fetch a cached result for a given task (e.g., classification, simplification).
    """
    with _LOCK:
        return _CACHE.get(f"{task}:{cache_key}")


def set_cached_result(task: str, cache_key: str, value: Any) -> None:
    """
    Store a result in the cache for reuse and persist it to disk.
    """
    with _LOCK:
        _CACHE[f"{task}:{cache_key}"] = value
        _save_cache_to_disk()


def clear_cache() -> None:
    """
    Utility to clear the cache (in-memory and on disk).
    """
    with _LOCK:
        _CACHE.clear()
        _save_cache_to_disk()


def compute_text_hash(text: str, model_name: str = "", prompt_version: str = "embedding_v1") -> str:
    """
    Hash helper for arbitrary text (e.g., embeddings).
    """
    key_material = "||".join([
        prompt_version,
        model_name or "",
        (text or "").strip()
    ])
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()
