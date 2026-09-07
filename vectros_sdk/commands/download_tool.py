import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DEFAULT_TOOLHUB_URL = "https://app.aios.foundation"


def download_tool(
    tool_author: str,
    tool_name: str,
    tool_version: Optional[str] = None,
    toolhub_url: str = DEFAULT_TOOLHUB_URL,
    target_dir: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Download a tool package from AIOS Tool Hub to local storage.

    Args:
        tool_author: Author identifier for the tool.
        tool_name: Name of the tool to download.
        tool_version: Optional version string.
        toolhub_url: Tool Hub endpoint URL.
        target_dir: Destination directory path on local machine.
        timeout: Request timeout in seconds.

    Returns:
        Dict[str, Any]: Operation status and target directory path.
    """
    base = toolhub_url.rstrip("/")
    params = {"author": tool_author, "name": tool_name}
    if tool_version:
        params["version"] = tool_version

    endpoint = f"{base}/tools/download"

    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
        if not response.ok:
            # Fallback to direct path endpoint
            alt_endpoint = f"{base}/tools/{tool_author}/{tool_name}"
            response = requests.get(alt_endpoint, params={"version": tool_version} if tool_version else None, timeout=timeout)

        if not response.ok:
            error_msg = f"Failed to download tool {tool_author}/{tool_name}: HTTP {response.status_code} - {response.text}"
            return {"success": False, "error": error_msg}

        payload = response.json()

        # Determine target destination directory
        if target_dir:
            dest_dir = Path(target_dir)
        else:
            core_dir = Path(__file__).parent.parent / "tool" / "core"
            dest_dir = core_dir / tool_author / tool_name

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json
        config_data = payload.get("config", {
            "name": tool_name,
            "description": payload.get("description", ""),
            "meta": {
                "author": tool_author,
                "version": tool_version or payload.get("version", "0.0.1"),
                "license": payload.get("license", "CC0"),
            },
            "build": payload.get("build", {
                "entry": "entry.py",
                "module": tool_name.capitalize(),
            }),
        })
        with open(dest_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        # Write entry.py or code files
        files = payload.get("files", {})
        if files and isinstance(files, dict):
            for rel_path, file_content in files.items():
                file_dest = dest_dir / rel_path
                file_dest.parent.mkdir(parents=True, exist_ok=True)
                with open(file_dest, "w", encoding="utf-8") as f:
                    f.write(file_content)
        elif "code" in payload:
            entry_filename = config_data.get("build", {}).get("entry", "entry.py")
            with open(dest_dir / entry_filename, "w", encoding="utf-8") as f:
                f.write(payload["code"])
        else:
            # Default stub entry if not provided in payload
            entry_filename = config_data.get("build", {}).get("entry", "entry.py")
            entry_path = dest_dir / entry_filename
            if not entry_path.exists():
                with open(entry_path, "w", encoding="utf-8") as f:
                    f.write(f"# Auto-generated tool entry for {tool_name}\n")

        return {
            "success": True,
            "tool_name": tool_name,
            "tool_author": tool_author,
            "target_dir": str(dest_dir),
        }

    except Exception as exc:
        return {"success": False, "error": f"Error downloading tool: {exc}"}


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download a tool from the AIOS Tool Hub.")
    parser.add_argument("--tool_author", required=True, help="Author of the tool")
    parser.add_argument("--tool_name", required=True, help="Name of the tool")
    parser.add_argument("--tool_version", default=None, help="Version of the tool (optional)")
    parser.add_argument("--toolhub_url", default=DEFAULT_TOOLHUB_URL, help=f"Tool Hub URL (default: {DEFAULT_TOOLHUB_URL})")
    parser.add_argument("--target_dir", default=None, help="Local directory to install the tool")

    parsed = parser.parse_args(args)
    res = download_tool(
        tool_author=parsed.tool_author,
        tool_name=parsed.tool_name,
        tool_version=parsed.tool_version,
        toolhub_url=parsed.toolhub_url,
        target_dir=parsed.target_dir,
    )

    if res.get("success"):
        print(f"Successfully downloaded tool '{parsed.tool_name}' by '{parsed.tool_author}' to {res.get('target_dir')}")
        return 0
    else:
        print(f"Failed to download tool: {res.get('error')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
