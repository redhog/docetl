"""
YAML path rewriter for the DocETL server.

When sending pipeline YAML to the client, all storage paths (absolute local
paths or remote fsspec URLs that live under DOCETL_STORAGE_URL) are rewritten
to HTTP URLs served by this server:

    /home/alice/.docetl/ns/files/foo.json
        → http://localhost:8000/files/ns/files/foo.json

When receiving YAML from the client, HTTP /files/… URLs are rewritten back
to full storage paths before handing off to the runner or writing to disk.

The rewriter walks the parsed YAML tree and rewrites any string value that
looks like a file path (absolute, fsspec URL, or already an /files/ HTTP URL).
Only values under known path keys are rewritten — this avoids mangling prompt
text that happens to contain slashes.
"""

import os
import re
from typing import Any
from urllib.parse import urlparse

import yaml

from docetl.storage import StorageBackend, get_default_backend

# Keys in pipeline YAML whose values are file paths
_PATH_KEYS = {
    # dataset input
    "path",
    # pipeline output
    "output_path",
    # pipeline output block
    "intermediate_dir",
    # top-level output path inside pipeline.output
    # (matched by key name "path" above, but also by positional detection)
}

# Regex: value looks like an absolute path or a known scheme URL
_ABS_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|[a-z][a-z0-9+\-.]*://)")
# HTTP /files/ prefix served by this server
_FILES_PREFIX = "/files/"


def _get_backend() -> StorageBackend:
    return get_default_backend()


def _server_base_url() -> str:
    """
    Return the base URL of this server, e.g. http://localhost:8000.
    Reads BACKEND_HOST / BACKEND_PORT env vars (same as main.py).
    """
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = os.getenv("BACKEND_PORT", "8000")
    scheme = os.getenv("BACKEND_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


# ---------------------------------------------------------------------------
# Low-level converters
# ---------------------------------------------------------------------------

def storage_path_to_http(path: str) -> str:
    """
    Convert an absolute/remote storage path to an HTTP /files/… URL.
    If the path is already an HTTP URL (not a /files/ URL on this server)
    it is returned unchanged.
    """
    backend = _get_backend()
    rel = backend.to_storage_relative(path)
    if rel is None:
        # Not under storage root — return as-is
        return path
    # Normalise separators to forward slashes
    rel = rel.replace("\\", "/").lstrip("/")
    return f"{_server_base_url()}{_FILES_PREFIX}{rel}"


def http_to_storage_path(url: str) -> str:
    """
    Convert an HTTP /files/… URL back to the full storage path.
    Other values are returned unchanged.
    """
    parsed = urlparse(url)
    if parsed.path.startswith(_FILES_PREFIX):
        rel = parsed.path[len(_FILES_PREFIX):]
        backend = _get_backend()
        return backend.from_storage_relative(rel)
    return url


def _is_storage_path(value: str) -> bool:
    """True if the string looks like an absolute/remote path (not already HTTP)."""
    if value.startswith(("http://", "https://")):
        # Only treat as storage if it is already a /files/ URL on this server
        parsed = urlparse(value)
        return parsed.path.startswith(_FILES_PREFIX)
    return bool(_ABS_RE.match(value))


def _is_files_url(value: str) -> bool:
    """True if the string is an HTTP /files/… URL (from this server)."""
    if not value.startswith(("http://", "https://")):
        return False
    parsed = urlparse(value)
    return parsed.path.startswith(_FILES_PREFIX)


# ---------------------------------------------------------------------------
# YAML tree walkers
# ---------------------------------------------------------------------------

def _rewrite_value_to_http(value: Any, key: str | None = None) -> Any:
    """Recursively rewrite storage paths → HTTP URLs in a parsed YAML tree."""
    if isinstance(value, dict):
        return {k: _rewrite_value_to_http(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_value_to_http(item, key=key) for item in value]
    if isinstance(value, str) and key in _PATH_KEYS and _is_storage_path(value):
        return storage_path_to_http(value)
    return value


def _rewrite_value_to_storage(value: Any, key: str | None = None) -> Any:
    """Recursively rewrite HTTP /files/ URLs → storage paths in a parsed YAML tree."""
    if isinstance(value, dict):
        return {k: _rewrite_value_to_storage(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_value_to_storage(item, key=key) for item in value]
    if isinstance(value, str) and key in _PATH_KEYS and _is_files_url(value):
        return http_to_storage_path(value)
    return value


# ---------------------------------------------------------------------------
# Public API: operate on YAML strings
# ---------------------------------------------------------------------------

def yaml_paths_to_http(yaml_str: str) -> str:
    """
    Parse a YAML pipeline config string, rewrite all storage paths to HTTP
    /files/… URLs, and return the re-serialised YAML string.
    """
    config = yaml.safe_load(yaml_str)
    if not isinstance(config, dict):
        return yaml_str
    rewritten = _rewrite_value_to_http(config)
    return yaml.dump(rewritten, allow_unicode=True, sort_keys=False)


def yaml_paths_to_storage(yaml_str: str) -> str:
    """
    Parse a YAML pipeline config string, rewrite all HTTP /files/… URLs back
    to storage paths, and return the re-serialised YAML string.
    """
    config = yaml.safe_load(yaml_str)
    if not isinstance(config, dict):
        return yaml_str
    rewritten = _rewrite_value_to_storage(config)
    return yaml.dump(rewritten, allow_unicode=True, sort_keys=False)


def rewrite_path_to_http(path: str) -> str:
    """Rewrite a single path string → HTTP URL (convenience wrapper)."""
    if _is_storage_path(path):
        return storage_path_to_http(path)
    return path


def rewrite_path_to_storage(path: str) -> str:
    """Rewrite a single path string → storage path (convenience wrapper)."""
    if _is_files_url(path):
        return http_to_storage_path(path)
    return path
