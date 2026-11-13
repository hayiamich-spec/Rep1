#!/usr/bin/env python3
"""Download one version-pinned public dataset file from Hugging Face Hub."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

MAX_ALLOWED_BYTES = 250 * 1024 * 1024
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,150}$")
FILE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_FIELDS = {"repo_id", "filename", "revision", "max_bytes"}

def parse_request(environ):
    raw = environ.get("HF_REQUEST_JSON", "").strip()
    if raw:
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Issue body must be a JSON object") from exc
    else:
        request = {
            "repo_id": environ.get("HF_INPUT_REPO_ID", ""),
            "filename": environ.get("HF_INPUT_FILENAME", ""),
            "revision": environ.get("HF_INPUT_REVISION", "main") or "main",
        }
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    if set(request) - ALLOWED_FIELDS:
        raise ValueError("Unexpected request fields")
    repo_id = request.get("repo_id", "")
    filename = request.get("filename", "")
    revision = request.get("revision", "main")
    max_bytes = request.get("max_bytes", DEFAULT_MAX_BYTES)
    if not isinstance(repo_id, str) or not REPO_PATTERN.fullmatch(repo_id):
        raise ValueError("repo_id must be an owner/dataset identifier")
    if not isinstance(filename, str) or not filename or len(filename) > 250:
        raise ValueError("Invalid filename")
    parts = filename.split("/")
    if any(part in ("", ".", "..") or not FILE_SEGMENT_PATTERN.fullmatch(part) for part in parts):
        raise ValueError("Unsafe filename; only relative dataset paths are accepted")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("Invalid revision")
    if ".." in revision.split("/"):
        raise ValueError("Invalid revision path")
    if type(max_bytes) is not int or not (1 <= max_bytes <= MAX_ALLOWED_BYTES):
        raise ValueError("max_bytes must be an integer from 1 to 262144000")
    return repo_id, filename, revision, max_bytes

def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    repo_id, filename, revision, max_bytes = parse_request(os.environ)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision, files_metadata=True)
    match = next((entry for entry in info.siblings or [] if entry.rfilename == filename), None)
    if match is None:
        raise ValueError(f"File not found in dataset revision: {filename}")
    size = match.size
    if not isinstance(size, int):
        raise ValueError("Remote file size is unknown: refusing unbounded download")
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes exceeds {max_bytes} byte budget")
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError("Could not resolve immutable dataset commit")
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename,
        repo_type="dataset", revision=resolved_revision))
    if downloaded.stat().st_size > max_bytes:
        raise RuntimeError("Downloaded file exceeds limit")
    output = Path("hf-output")
    output.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".bin"
    destination = output / f"dataset_file{suffix}"
    shutil.copy2(downloaded, destination)
    manifest = {
        "source": "huggingface", "repo_type": "dataset", "repo_id": repo_id,
        "filename": filename, "requested_revision": revision,
        "resolved_commit": resolved_revision,
        "file_size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination), "artifact_file": destination.name,
        "note": "Raw source file. Not a TAU training example or trained model."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"HF import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
