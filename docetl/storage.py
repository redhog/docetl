"""
StorageBackend: fsspec-backed storage with local forever-cache.

Content-addressed filenames are valid indefinitely — no TTL, no invalidation.
For mutable files (workspace.yaml), use bypass_cache=True to skip local cache.

Configure via environment variables:
    DOCETL_STORAGE_URL   — default: ~/.docetl  (any fsspec URL)
    DOCETL_CACHE_DIR     — default: ~/.docetl_cache
"""

import hashlib
import io
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

import fsspec


def _default_storage_url() -> str:
    home = os.getenv("DOCETL_HOME_DIR", os.path.expanduser("~"))
    return os.getenv("DOCETL_STORAGE_URL", os.path.join(home, ".docetl"))


def _default_cache_dir() -> str:
    return os.getenv(
        "DOCETL_CACHE_DIR",
        os.path.expanduser("~/.docetl_cache"),
    )


class _Writethrough:
    """
    File-like object that writes to a local temp file and pushes to remote on close.
    """

    def __init__(self, backend: "StorageBackend", remote_path: str, mode: str, **kw):
        self._backend = backend
        self._remote_path = remote_path
        self._mode = mode
        self._tmp = tempfile.NamedTemporaryFile(
            mode=mode, delete=False, suffix=".docetl_tmp"
        )

    # Delegate file-like interface to the temp file
    def write(self, data):
        return self._tmp.write(data)

    def writelines(self, lines):
        return self._tmp.writelines(lines)

    def flush(self):
        return self._tmp.flush()

    def tell(self):
        return self._tmp.tell()

    def seek(self, *args):
        return self._tmp.seek(*args)

    @property
    def name(self):
        return self._tmp.name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if not self._tmp.closed:
            self._tmp.flush()
            self._tmp.close()
            self._backend._push(self._tmp.name, self._remote_path)
            os.unlink(self._tmp.name)


class StorageBackend:
    """
    fsspec-backed storage with local forever-cache keyed by content-addressed filenames.

    API mirrors fsspec AbstractFileSystem:
      open(path, mode, **kw) -> file-like
      exists(path) -> bool
      makedirs(path, exist_ok=False)
      rm(path, recursive=False)
      ls(path) -> list[str]
      info(path) -> dict
      copy(path1, path2)
      mv(path1, path2)
      glob(pattern) -> list[str]
      get(rpath, lpath, recursive=False)
      put(lpath, rpath, recursive=False)

    Additional:
      open(..., bypass_cache=True) — always fetch from remote (mutable files like workspace.yaml)
    """

    def __init__(
        self,
        remote_root: str | None = None,
        local_cache_dir: str | None = None,
    ):
        self._remote_root = remote_root or _default_storage_url()
        self.local_cache_dir = local_cache_dir or _default_cache_dir()
        self.fs, self._fs_root = fsspec.url_to_fs(self._remote_root)

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _remote_path(self, path: str) -> str:
        """Resolve a relative-or-absolute path to the remote (fsspec) path."""
        path = path.rstrip("/")
        if os.path.isabs(path) or "://" in path:
            return path
        return self.fs.sep.join([self._fs_root.rstrip(self.fs.sep), path])

    def _local_cache_path(self, remote_path: str) -> str:
        """Mirror a remote path under local_cache_dir."""
        # Strip scheme and leading slashes for local mirror path
        if "://" in remote_path:
            _, rest = remote_path.split("://", 1)
        else:
            rest = remote_path
        return os.path.join(self.local_cache_dir, rest.lstrip("/"))

    def _is_local(self) -> bool:
        return isinstance(self.fs, fsspec.implementations.local.LocalFileSystem)

    # ------------------------------------------------------------------
    # Push / pull helpers
    # ------------------------------------------------------------------

    def _push(self, local_path: str, remote_path: str) -> None:
        """Copy local file to remote. No-op when remote IS local."""
        rpath = self._remote_path(remote_path)
        if self._is_local() and os.path.abspath(local_path) == os.path.abspath(rpath):
            return
        parent = self.fs.sep.join(rpath.split(self.fs.sep)[:-1])
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.put_file(local_path, rpath)

    def _pull(self, remote_path: str, local_path: str) -> None:
        """Copy remote file to local cache path."""
        rpath = self._remote_path(remote_path)
        if self._is_local() and os.path.abspath(rpath) == os.path.abspath(local_path):
            return
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.fs.get_file(rpath, local_path)

    # ------------------------------------------------------------------
    # Core API (mirrors fsspec)
    # ------------------------------------------------------------------

    def open(
        self,
        path: str,
        mode: str = "r",
        bypass_cache: bool = False,
        **kw,
    ):
        """
        Open a file for reading or writing.

        Read path:
          1. If not bypass_cache and local cache exists → return cached file.
          2. Otherwise pull from remote, store in cache, return cached file.
        Write path:
          Write to a temp file; on close, push to remote (and update cache).
        bypass_cache:
          Always fetch from remote; do not read or write local cache.
          Use for mutable files like workspace.yaml.
        """
        rpath = self._remote_path(path)
        lpath = self._local_cache_path(rpath)

        if "r" in mode or mode == "rb":
            if not bypass_cache and os.path.exists(lpath):
                return open(lpath, mode, **kw)
            # Pull from remote
            self._pull(rpath, lpath)
            if bypass_cache:
                # Return a in-memory buffer so we don't pollute the cache
                with open(lpath, mode, **kw) as f:
                    data = f.read()
                os.unlink(lpath)
                if "b" in mode:
                    return io.BytesIO(data)
                return io.StringIO(data)
            return open(lpath, mode, **kw)

        # Write mode
        if bypass_cache:
            # Write directly to remote via fsspec
            parent = self.fs.sep.join(rpath.split(self.fs.sep)[:-1])
            if parent:
                self.fs.makedirs(parent, exist_ok=True)
            return self.fs.open(rpath, mode, **kw)

        return _Writethrough(self, path, mode, **kw)

    def exists(self, path: str) -> bool:
        """Check existence on remote (authoritative)."""
        return self.fs.exists(self._remote_path(path))

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        self.fs.makedirs(self._remote_path(path), exist_ok=exist_ok)

    def rm(self, path: str, recursive: bool = False) -> None:
        rpath = self._remote_path(path)
        self.fs.rm(rpath, recursive=recursive)
        # Also clean local cache
        lpath = self._local_cache_path(rpath)
        if os.path.isdir(lpath) and recursive:
            shutil.rmtree(lpath, ignore_errors=True)
        elif os.path.exists(lpath):
            os.unlink(lpath)

    def ls(self, path: str, detail: bool = False) -> list:
        return self.fs.ls(self._remote_path(path), detail=detail)

    def info(self, path: str) -> dict:
        return self.fs.info(self._remote_path(path))

    def copy(self, path1: str, path2: str) -> None:
        self.fs.copy(self._remote_path(path1), self._remote_path(path2))

    def mv(self, path1: str, path2: str) -> None:
        self.fs.mv(self._remote_path(path1), self._remote_path(path2))

    def glob(self, pattern: str) -> list[str]:
        return self.fs.glob(self._remote_path(pattern))

    def get(self, rpath: str, lpath: str, recursive: bool = False) -> None:
        self.fs.get(self._remote_path(rpath), lpath, recursive=recursive)

    def put(self, lpath: str, rpath: str, recursive: bool = False) -> None:
        self.fs.put(lpath, self._remote_path(rpath), recursive=recursive)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def isdir(self, path: str) -> bool:
        return self.fs.isdir(self._remote_path(path))

    def isfile(self, path: str) -> bool:
        return self.fs.isfile(self._remote_path(path))

    # ------------------------------------------------------------------
    # Storage-relative path helpers (used by HTTP serving layer)
    # ------------------------------------------------------------------

    def _scheme(self) -> str:
        """Return the URL scheme of the remote root (e.g. 'gs', 's3', '' for local)."""
        if "://" in self._remote_root:
            return self._remote_root.split("://", 1)[0]
        return ""

    def to_storage_relative(self, path: str) -> str | None:
        """
        Given an absolute/fsspec path, return the path relative to _fs_root,
        or None if it does not live under the storage root.

        Example (local):
            _fs_root = /home/alice/.docetl
            path     = /home/alice/.docetl/ns/files/foo.json
            returns  = ns/files/foo.json

        Example (gcs):
            _remote_root = gs://my-bucket/docetl
            _fs_root     = my-bucket/docetl
            path         = gs://my-bucket/docetl/ns/files/foo.json
            returns      = ns/files/foo.json
        """
        # Normalise: strip scheme so we can compare against _fs_root (which has no scheme)
        scheme = self._scheme()
        if scheme and path.startswith(f"{scheme}://"):
            path_no_scheme = path[len(scheme) + 3:]
        else:
            path_no_scheme = path

        # rpath via _remote_path may reattach scheme; normalise that too
        rpath = self._remote_path(path)
        if scheme and rpath.startswith(f"{scheme}://"):
            rpath_no_scheme = rpath[len(scheme) + 3:]
        else:
            rpath_no_scheme = rpath

        root = self._fs_root.rstrip(self.fs.sep) + self.fs.sep
        if rpath_no_scheme.startswith(root):
            return rpath_no_scheme[len(root):]
        # Also handle if path was already relative (no change needed)
        if not os.path.isabs(path_no_scheme) and "://" not in path:
            return path_no_scheme.lstrip("/")
        return None

    def from_storage_relative(self, rel: str) -> str:
        """
        Given a path relative to the storage root, return the full fsspec path
        (including scheme for remote filesystems).

        Example (local):
            rel     = ns/files/foo.json
            returns = /home/alice/.docetl/ns/files/foo.json

        Example (gcs):
            _remote_root = gs://my-bucket/docetl
            rel          = ns/files/foo.json
            returns      = gs://my-bucket/docetl/ns/files/foo.json
        """
        scheme = self._scheme()
        root = self._fs_root.rstrip(self.fs.sep)
        path = self.fs.sep.join([root, rel.lstrip("/")])
        if scheme:
            return f"{scheme}://{path}"
        return path


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised)
# ---------------------------------------------------------------------------

_default_backend: StorageBackend | None = None


def get_default_backend() -> StorageBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = StorageBackend()
    return _default_backend


# ---------------------------------------------------------------------------
# Content-address helpers (used by runner / server)
# ---------------------------------------------------------------------------

def content_hash(data: bytes | str) -> str:
    """SHA-256 hex digest of bytes or UTF-8 string."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(obj: Any) -> str:
    """Stable SHA-256 hex of a JSON-serialisable object."""
    import json

    serialised = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return content_hash(serialised)


def storage_path_join(*parts: str) -> str:
    """
    Join path segments for any fsspec URL, including s3://, http://, local.
    Uses forward slashes throughout; does NOT use os.path.join (which mangles
    s3:// → s3:/ on some platforms).
    """
    if not parts:
        return ""
    base = parts[0].rstrip("/")
    for part in parts[1:]:
        base = base + "/" + part.strip("/")
    return base
