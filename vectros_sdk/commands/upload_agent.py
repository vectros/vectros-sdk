import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DEFAULT_AGENTHUB_URL = "https://app.aios.foundation"


def upload_agent(
    agent_path: str,
    agenthub_url: str = DEFAULT_AGENTHUB_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Publish a custom agent package to the AIOS Agent Hub.

    Args:
        agent_path: Path to the agent directory containing config.json and entry file.
        agenthub_url: Agent Hub endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict[str, Any]: Upload status result from the Agent Hub.
    """
    path = Path(agent_path)
    if not path.exists() or not path.is_dir():
        return {"success": False, "error": f"Agent directory not found at: {agent_path}"}

    config_file = path / "config.json"
    if not config_file.exists():
        return {"success": False, "error": f"Missing config.json in agent directory: {agent_path}"}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        return {"success": False, "error": f"Failed to parse config.json: {exc}"}

    # Validate required metadata
    agent_name = config.get("name")
    if not agent_name:
        return {"success": False, "error": "Invalid config.json: missing 'name' field"}

    build_info = config.get("build", {})
    entry_file_name = build_info.get("entry", "entry.py")
    entry_path = path / entry_file_name
    if not entry_path.exists():
        return {"success": False, "error": f"Missing entry file '{entry_file_name}' in {agent_path}"}

    try:
        with open(entry_path, "r", encoding="utf-8") as f:
            code_content = f.read()
    except Exception as exc:
        return {"success": False, "error": f"Failed to read entry file '{entry_file_name}': {exc}"}

    # Collect all other files in directory
    files_payload = {}
    for p in path.rglob("*"):
        if p.is_file() and p != config_file and not p.name.endswith(".pyc") and "__pycache__" not in p.parts:
            rel_name = str(p.relative_to(path))
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    files_payload[rel_name] = f.read()
            except Exception:
                pass

    upload_payload = {
        "name": agent_name,
        "config": config,
        "code": code_content,
        "files": files_payload,
    }

    base = agenthub_url.rstrip("/")
    endpoint = f"{base}/agents/upload"

    try:
        response = requests.post(
            endpoint,
            json=upload_payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.ok:
            try:
                res_data = response.json()
            except Exception:
                res_data = {"message": response.text}
            return {"success": True, "agent_name": agent_name, "response": res_data}

        return {
            "success": False,
            "error": f"Agent Hub returned status {response.status_code}: {response.text}",
        }
    except Exception as exc:
        return {"success": False, "error": f"Error uploading agent to Agent Hub: {exc}"}


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload and share custom agent with AIOS Agent Hub.")
    parser.add_argument("--agent_path", required=True, help="Path to the agent directory")
    parser.add_argument("--agenthub_url", default=DEFAULT_AGENTHUB_URL, help=f"Agent Hub URL (default: {DEFAULT_AGENTHUB_URL})")

    parsed = parser.parse_args(args)
    res = upload_agent(agent_path=parsed.agent_path, agenthub_url=parsed.agenthub_url)

    if res.get("success"):
        print(f"Successfully uploaded agent from '{parsed.agent_path}' to Agent Hub ({parsed.agenthub_url})")
        return 0
    else:
        print(f"Failed to upload agent: {res.get('error')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
