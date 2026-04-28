from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
import os
import yaml
import httpx
import json
import csv
from io import StringIO
from server.app.models import PipelineConfigRequest, WorkspaceSaveRequest
from docetl.storage import StorageBackend, get_default_backend
from server.app.path_rewriter import (
    yaml_paths_to_http,
    yaml_paths_to_storage,
    rewrite_path_to_http,
    rewrite_path_to_storage,
    http_to_storage_path,
    storage_path_to_http,
)

router = APIRouter()


def _storage() -> StorageBackend:
    return get_default_backend()


def _namespace_dir(namespace: str) -> str:
    """Namespace directory path relative to storage root, e.g. 'ns-uuid'."""
    return namespace


# ---------------------------------------------------------------------------
# Storage-backed file server: GET /files/{rel_path}
# ---------------------------------------------------------------------------

@router.get("/files/{rel_path:path}")
async def serve_storage_file(rel_path: str, request: Request):
    """
    Serve any file from the storage backend by its storage-relative path.
    Content-addressed files (pipeline configs, checkpoints, uploads) are
    cached forever; mutable files (workspace.yaml) must be accessed via
    the /fs/workspace/{id} endpoint.

    This endpoint is the target of all /files/… URLs embedded in pipeline YAML.
    """
    try:
        if ".." in rel_path:
            raise HTTPException(status_code=400, detail="Invalid path")

        backend = _storage()
        storage_path = backend.from_storage_relative(rel_path)

        if not backend.exists(storage_path):
            raise HTTPException(status_code=404, detail="File not found")

        # If the local cache already has the file, serve it directly (fast path)
        local_cache = backend._local_cache_path(storage_path)
        if os.path.isfile(local_cache):
            return FileResponse(
                path=local_cache,
                filename=os.path.basename(rel_path),
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

        # Otherwise stream through the storage backend (pulls to cache on the way)
        def _iter():
            with backend.open(storage_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            _iter(),
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

@router.post("/fs/check-namespace")
async def check_namespace(namespace: str):
    """Check if namespace exists and create if it doesn't"""
    try:
        backend = _storage()
        ns_dir = _namespace_dir(namespace)
        exists = backend.exists(ns_dir)
        if not exists:
            backend.makedirs(ns_dir, exist_ok=True)
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check/create namespace: {str(e)}")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

def validate_json_content(content: bytes) -> None:
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


def convert_csv_to_json(csv_content: bytes) -> bytes:
    try:
        csv_string = csv_content.decode('utf-8')
        reader = csv.DictReader(StringIO(csv_string))
        data = list(reader)
        if not data:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        return json.dumps(data).encode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV encoding")
    except csv.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")


def is_likely_csv(content: bytes, filename: str) -> bool:
    if filename.lower().endswith('.csv'):
        return True
    try:
        first_line = content.split(b'\n')[0].decode('utf-8')
        return ',' in first_line and not any(c in first_line for c in '{}[]')
    except Exception:
        return False


@router.post("/fs/upload-file")
async def upload_file(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    namespace: str = Form(...)
):
    """Upload a file; returns HTTP /files/… URL for the stored file."""
    try:
        if not file and not url:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        backend = _storage()
        upload_dir = f"{_namespace_dir(namespace)}/files"
        backend.makedirs(upload_dir, exist_ok=True)

        if url:
            filename = url.split("/")[-1] or "dataset.json"
            storage_path = f"{upload_dir}/{filename.replace('.csv', '.json')}"

            async with httpx.AsyncClient() as client:
                async with client.stream('GET', url, follow_redirects=True) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download from URL: {response.status_code}"
                        )
                    content_chunks = []
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if chunk:
                            content_chunks.append(chunk)
                    content = b''.join(content_chunks)

            if is_likely_csv(content, filename):
                content = convert_csv_to_json(content)
            validate_json_content(content)

            with backend.open(storage_path, "wb") as f:
                f.write(content)
        else:
            file_content = await file.read()
            if file.filename.lower().endswith('.csv'):
                file_content = convert_csv_to_json(file_content)
            validate_json_content(file_content)
            storage_path = f"{upload_dir}/{file.filename.replace('.csv', '.json')}"
            with backend.open(storage_path, "wb") as f:
                f.write(file_content)

        # Return HTTP URL so client never sees raw storage paths
        http_url = storage_path_to_http(backend.from_storage_relative(storage_path))
        return {"path": http_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/fs/save-documents")
async def save_documents(files: list[UploadFile] = File(...), namespace: str = Form(...)):
    """Save documents; returns HTTP /files/… URLs."""
    try:
        backend = _storage()
        uploads_dir = f"{_namespace_dir(namespace)}/documents"
        backend.makedirs(uploads_dir, exist_ok=True)

        saved_files = []
        for file in files:
            safe_name = "".join(c if c.isalnum() or c in ".-" else "_" for c in file.filename)
            storage_path = f"{uploads_dir}/{safe_name}"
            content = await file.read()
            with backend.open(storage_path, "wb") as f:
                f.write(content)
            http_url = storage_path_to_http(backend.from_storage_relative(storage_path))
            saved_files.append({"name": file.filename, "path": http_url})

        return {"files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save documents: {str(e)}")


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@router.post("/fs/write-pipeline-config")
async def write_pipeline_config(request: PipelineConfigRequest):
    """
    Receive pipeline YAML from client (paths as HTTP /files/… URLs),
    rewrite to storage paths, write to storage, return HTTP URLs.
    """
    try:
        backend = _storage()

        # Rewrite HTTP /files/ URLs → storage paths before persisting
        storage_yaml = yaml_paths_to_storage(request.config)

        pipeline_dir = f"{_namespace_dir(request.namespace)}/pipelines"
        config_dir = f"{pipeline_dir}/configs"
        name_dir = f"{pipeline_dir}/{request.name}/intermediates"

        backend.makedirs(config_dir, exist_ok=True)
        backend.makedirs(name_dir, exist_ok=True)

        storage_path = f"{config_dir}/{request.name}.yaml"
        with backend.open(storage_path, "w") as f:
            f.write(storage_yaml)

        # Rewrite response paths → HTTP URLs for the client
        http_file_path = storage_path_to_http(backend.from_storage_relative(storage_path))
        http_input_path = rewrite_path_to_http(request.input_path) if request.input_path else request.input_path
        http_output_path = rewrite_path_to_http(request.output_path) if request.output_path else request.output_path

        return {
            "filePath": http_file_path,
            "inputPath": http_input_path,
            "outputPath": http_output_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write pipeline configuration: {str(e)}")


# ---------------------------------------------------------------------------
# Generic file reads (legacy endpoints — accept storage paths or HTTP URLs)
# ---------------------------------------------------------------------------

def _resolve_to_storage_path(path: str) -> str:
    """Accept a storage path or HTTP /files/ URL; return storage path."""
    from server.app.path_rewriter import _is_files_url
    if _is_files_url(path):
        return http_to_storage_path(path)
    return path


@router.get("/fs/read-file")
async def read_file(path: str):
    """Read file contents. Accepts storage paths or HTTP /files/ URLs.
    When serving YAML, rewrites embedded paths to HTTP URLs."""
    try:
        if path.startswith(("http://", "https://")) and not path.split("//", 1)[1].startswith(
            ("localhost", "127.0.0.1")
        ):
            raise HTTPException(status_code=400, detail="External HTTP URLs not supported")

        storage_path = _resolve_to_storage_path(path)
        backend = _storage()
        if not backend.exists(storage_path):
            raise HTTPException(status_code=404, detail="File not found")

        # For YAML files: read, rewrite paths, return as text
        if storage_path.endswith((".yaml", ".yml")):
            with backend.open(storage_path, "r", bypass_cache=storage_path.endswith("workspace.yaml")) as f:
                raw = f.read()
            rewritten = yaml_paths_to_http(raw)
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(rewritten, media_type="text/yaml")

        # Fast path: serve from local cache if available
        local_cache = backend._local_cache_path(backend._remote_path(storage_path))
        if os.path.isfile(local_cache):
            return FileResponse(local_cache)

        def _iter():
            with backend.open(storage_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(_iter())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/fs/read-file-page")
async def read_file_page(path: str, page: int = 0, chunk_size: int = 500000):
    """Paginated file read. Accepts storage paths or HTTP /files/ URLs."""
    try:
        storage_path = _resolve_to_storage_path(path)
        backend = _storage()
        if not backend.exists(storage_path):
            raise HTTPException(status_code=404, detail="File not found")

        file_size = backend.info(storage_path).get("size", 0)
        start = page * chunk_size

        with backend.open(storage_path, "rb") as f:
            f.seek(start)
            content = f.read(chunk_size).decode("utf-8")

        return {
            "content": content,
            "totalSize": file_size,
            "page": page,
            "hasMore": start + len(content) < file_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/fs/serve-document/{path:path}")
async def serve_document(path: str):
    """Serve document files by storage-relative path."""
    try:
        if ".." in path:
            raise HTTPException(status_code=400, detail="Invalid file path")

        backend = _storage()
        storage_path = backend.from_storage_relative(path)
        if not backend.exists(storage_path):
            raise HTTPException(status_code=404, detail="File not found")

        local_cache = backend._local_cache_path(storage_path)
        if os.path.isfile(local_cache):
            return FileResponse(
                path=local_cache,
                filename=os.path.basename(path),
                headers={"Cache-Control": "public, max-age=3600"},
            )

        def _iter():
            with backend.open(storage_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            _iter(),
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


# ---------------------------------------------------------------------------
# Workspace (mutable — bypass cache, paths not content-addressed)
# ---------------------------------------------------------------------------

@router.get("/fs/workspace/{workspace_id}")
async def load_workspace(workspace_id: str):
    """Load workspace YAML; rewrites embedded paths to HTTP URLs."""
    try:
        backend = _storage()
        workspace_rel = f"{_namespace_dir(workspace_id)}/workspace.yaml"
        storage_path = backend.from_storage_relative(workspace_rel)
        if not backend.exists(storage_path):
            raise HTTPException(status_code=404, detail="Workspace not found")
        with backend.open(storage_path, "r", bypass_cache=True) as f:
            raw = f.read()
        # workspace YAML may contain pipeline paths — rewrite for client
        content = yaml_paths_to_http(raw)
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load workspace: {str(e)}")


@router.post("/fs/workspace/{workspace_id}")
async def save_workspace(workspace_id: str, request: WorkspaceSaveRequest):
    """Save workspace YAML; rewrites HTTP /files/ URLs back to storage paths."""
    try:
        backend = _storage()
        ns_rel = _namespace_dir(workspace_id)
        backend.makedirs(ns_rel, exist_ok=True)
        workspace_rel = f"{ns_rel}/workspace.yaml"
        storage_path = backend.from_storage_relative(workspace_rel)
        storage_content = yaml_paths_to_storage(request.content)
        with backend.open(storage_path, "w", bypass_cache=True) as f:
            f.write(storage_content)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save workspace: {str(e)}")


@router.get("/fs/check-file")
async def check_file(path: str):
    """Check file existence. Accepts storage paths or HTTP /files/ URLs."""
    try:
        storage_path = _resolve_to_storage_path(path)
        backend = _storage()
        exists = backend.exists(storage_path)
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check file: {str(e)}")
