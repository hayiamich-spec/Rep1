#!/usr/bin/env python3
"""Delete a verified, previously downloaded HF-import workflow artifact."""
import json
import os
import re
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

def parse_request(env):
    raw = env.get("HF_CLEAN_JSON", "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Cleanup issue must have a JSON body") from exc
        if type(payload) is not dict or set(payload) != {"artifact_id"}:
            raise ValueError("Expected only artifact_id")
        artifact_id = payload["artifact_id"]
    else:
        artifact_id = env.get("HF_CLEAN_ARTIFACT_ID", "").strip()
        if isinstance(artifact_id, str) and artifact_id.isdecimal():
            artifact_id = int(artifact_id)
    if type(artifact_id) is not int or artifact_id < 1:
        raise ValueError("artifact_id must be a positive integer")
    return artifact_id

def validate_artifact(payload, expected_id):
    if type(payload) is not dict or payload.get("id") != expected_id:
        raise ValueError("Unexpected artifact ID")
    run = payload.get("workflow_run")
    if not isinstance(run, dict) or type(run.get("id")) is not int:
        raise ValueError("Missing source workflow run")
    if payload.get("name") != f"hf-dataset-{run['id']}":
        raise ValueError("Not a recognized HF importer artifact")

def request_json(url, token, method="GET"):
    request = Request(url, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hf-importer-cleanup",
    })
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read()
            return json.loads(body) if body else None
    except HTTPError as exc:
        raise RuntimeError(f"GitHub API {method} failed: HTTP {exc.code}") from exc

def main():
    artifact_id = parse_request(os.environ)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not token:
        raise ValueError("GitHub Actions repository and token are required")
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}"
    metadata = request_json(url, token)
    validate_artifact(metadata, artifact_id)
    request_json(url, token, method="DELETE")
    print(f"Deleted verified importer artifact ID {artifact_id} from {repo}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
