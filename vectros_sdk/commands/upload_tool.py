import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DEFAULT_TOOLHUB_URL = "https://app.aios.foundation"


def upload_tool(
    tool_path: str,
    toolhub_url: str = DEFAULT_TOOLHUB_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Publish a custom tool to the AIOS Tool Hub.

    Args:
        tool_path: Path to the tool directory containing config.json and entry file.
        toolhub_url: Tool Hub endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict[str, Any]: Upload status result from the Tool Hub.
    """
    path = Path(tool_path)
    if not path.exists() or not path.is_dir():
        return {"success": False, "error": f"Tool directory not found at: {tool_path}"}

    config_file = path / "config.json"
    if not config_file.exists():
        return {"success": False, "error": f"Missing config.json in tool directory: {tool_path}"}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        return {"success": False, "error": f"Failed to parse config.json: {exc}"}

    # Validate required metadata
    tool_name = config.get("name")
    if not tool_name:
        return {"success": False, "error": "Invalid config.json: missing 'name' field"}

    build_info = config.get("build", {})
    entry_file_name = build_info.get("entry", "entry.py")
    entry_path = path / entry_file_name
    if not entry_path.exists():
        return {"success": False, "error": f"Missing entry file '{entry_file_name}' in {tool_path}"}

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
        "name": tool_name,
        "config": config,
        "code": code_content,
        "files": files_payload,
    }

    base = toolhub_url.rstrip("/")
    endpoint = f"{base}/tools/upload"

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
            return {"success": True, "tool_name": tool_name, "response": res_data}

        return {
            "success": False,
            "error": f"Tool Hub returned status {response.status_code}: {response.text}",
        }
    except Exception as exc:
        return {"success": False, "error": f"Error uploading tool to Tool Hub: {exc}"}


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload and share custom tool with AIOS Tool Hub.")
    parser.add_argument("--tool_path", required=True, help="Path to the tool directory")
    parser.add_argument("--toolhub_url", default=DEFAULT_TOOLHUB_URL, help=f"Tool Hub URL (default: {DEFAULT_TOOLHUB_URL})")

    parsed = parser.parse_args(args)
    res = upload_tool(tool_path=parsed.tool_path, toolhub_url=parsed.toolhub_url)

    if res.get("success"):
        print(f"Successfully uploaded tool from '{parsed.tool_path}' to Tool Hub ({parsed.toolhub_url})")
        return 0
    else:
        print(f"Failed to upload tool: {res.get('error')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
