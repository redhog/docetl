from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os
import yaml
import shutil
import httpx
import json
import csv
from io import StringIO
from pathlib import Path
from server.app.models import PipelineConfigRequest, WorkspaceSaveRequest
from docetl.storage import StorageBackend, get_default_backend

router = APIRouter()


def get_home_dir() -> str:
    """Get the home directory from env var or user home"""
    return os.getenv("DOCETL_HOME_DIR", os.path.expanduser("~"))


def get_namespace_dir(namespace: str) -> str:
    """Get the namespace directory path (as str for StorageBackend)"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, ".docetl", namespace)


def _storage() -> StorageBackend:
    return get_default_backend()


@router.post("/check-namespace")
async def check_namespace(namespace: str):
    """Check if namespace exists and create if it doesn't"""
    try:
        backend = _storage()
        ns_dir = get_namespace_dir(namespace)
        exists = backend.exists(ns_dir)

        if not exists:
            backend.makedirs(ns_dir, exist_ok=True)

        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check/create namespace: {str(e)}")


def validate_json_content(content: bytes) -> None:
    """Validate that content can be parsed as JSON"""
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


def convert_csv_to_json(csv_content: bytes) -> bytes:
    """Convert CSV content to JSON format"""
    try:
        csv_string = csv_content.decode('utf-8')
        csv_file = StringIO(csv_string)
        reader = csv.DictReader(csv_file)
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
    except:
        return False


@router.post("/upload-file")
async def upload_file(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    namespace: str = Form(...)
):
    """Upload a file to the namespace files directory, either from a direct upload or a URL"""
    try:
        if not file and not url:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        backend = _storage()
        upload_dir = os.path.join(get_namespace_dir(namespace), "files")
        backend.makedirs(upload_dir, exist_ok=True)

        if url:
            filename = url.split("/")[-1] or "dataset.json"
            file_path = os.path.join(upload_dir, filename.replace('.csv', '.json'))

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
                        try:
                            content = convert_csv_to_json(content)
                        except HTTPException as e:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to convert CSV to JSON: {str(e.detail)}"
                            )

                    validate_json_content(content)

                    with backend.open(file_path, "wb") as f:
                        f.write(content)
        else:
            file_content = await file.read()

            if file.filename.lower().endswith('.csv'):
                try:
                    file_content = convert_csv_to_json(file_content)
                except HTTPException as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to convert CSV to JSON: {str(e.detail)}"
                    )

            validate_json_content(file_content)

            file_path = os.path.join(upload_dir, file.filename.replace('.csv', '.json'))
            with backend.open(file_path, "wb") as f:
                f.write(file_content)

        return {"path": file_path}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/save-documents")
async def save_documents(files: list[UploadFile] = File(...), namespace: str = Form(...)):
    """Save multiple documents to the namespace documents directory"""
    try:
        backend = _storage()
        uploads_dir = os.path.join(get_namespace_dir(namespace), "documents")
        backend.makedirs(uploads_dir, exist_ok=True)

        saved_files = []
        for file in files:
            safe_name = "".join(c if c.isalnum() or c in ".-" else "_" for c in file.filename)
            file_path = os.path.join(uploads_dir, safe_name)
            content = await file.read()
            with backend.open(file_path, "wb") as f:
                f.write(content)
            saved_files.append({"name": file.filename, "path": file_path})

        return {"files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save documents: {str(e)}")


@router.post("/write-pipeline-config")
async def write_pipeline_config(request: PipelineConfigRequest):
    """Write pipeline configuration YAML file"""
    try:
        backend = _storage()
        home_dir = get_home_dir()
        pipeline_dir = os.path.join(home_dir, ".docetl", request.namespace, "pipelines")
        config_dir = os.path.join(pipeline_dir, "configs")
        name_dir = os.path.join(pipeline_dir, request.name, "intermediates")

        backend.makedirs(config_dir, exist_ok=True)
        backend.makedirs(name_dir, exist_ok=True)

        file_path = os.path.join(config_dir, f"{request.name}.yaml")
        with backend.open(file_path, "w") as f:
            f.write(request.config)

        return {
            "filePath": file_path,
            "inputPath": request.input_path,
            "outputPath": request.output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write pipeline configuration: {str(e)}")


@router.get("/read-file")
async def read_file(path: str):
    """Read file contents"""
    try:
        if path.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="HTTP URLs not supported in this endpoint")

        backend = _storage()
        if not backend.exists(path):
            raise HTTPException(status_code=404, detail="File not found")

        # For local backend, FileResponse is most efficient
        if os.path.isfile(path):
            return FileResponse(path)

        # Remote backend: stream through
        def _iter():
            with backend.open(path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(_iter())
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/read-file-page")
async def read_file_page(path: str, page: int = 0, chunk_size: int = 500000):
    """Read file contents by page"""
    try:
        backend = _storage()
        if not backend.exists(path):
            raise HTTPException(status_code=404, detail="File not found")

        file_size = backend.info(path).get("size", 0)
        start = page * chunk_size

        with backend.open(path, "rb") as f:
            f.seek(start)
            content = f.read(chunk_size).decode("utf-8")

        return {
            "content": content,
            "totalSize": file_size,
            "page": page,
            "hasMore": start + len(content) < file_size
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/serve-document/{path:path}")
async def serve_document(path: str):
    """Serve document files"""
    try:
        if ".." in path:
            raise HTTPException(status_code=400, detail="Invalid file path")

        backend = _storage()
        if not backend.exists(path):
            raise HTTPException(status_code=404, detail="File not found")

        if os.path.isfile(path):
            return FileResponse(
                path=path,
                filename=os.path.basename(path),
                headers={"Cache-Control": "public, max-age=3600"}
            )

        def _iter():
            with backend.open(path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            _iter(),
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.get("/workspace/{workspace_id}")
async def load_workspace(workspace_id: str):
    """Load workspace state YAML for a given workspace UUID"""
    try:
        backend = _storage()
        workspace_file = os.path.join(get_namespace_dir(workspace_id), "workspace.yaml")
        if not backend.exists(workspace_file):
            raise HTTPException(status_code=404, detail="Workspace not found")
        # bypass_cache=True: workspace is mutable state
        with backend.open(workspace_file, "r", bypass_cache=True) as f:
            content = f.read()
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load workspace: {str(e)}")


@router.post("/workspace/{workspace_id}")
async def save_workspace(workspace_id: str, request: WorkspaceSaveRequest):
    """Save workspace state YAML for a given workspace UUID"""
    try:
        backend = _storage()
        ns_dir = get_namespace_dir(workspace_id)
        backend.makedirs(ns_dir, exist_ok=True)
        workspace_file = os.path.join(ns_dir, "workspace.yaml")
        # bypass_cache=True: mutable file — write directly through to remote
        with backend.open(workspace_file, "w", bypass_cache=True) as f:
            f.write(request.content)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save workspace: {str(e)}")


@router.get("/check-file")
async def check_file(path: str):
    """Check if a file exists without reading it"""
    try:
        backend = _storage()
        exists = backend.exists(path)
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check file: {str(e)}")
