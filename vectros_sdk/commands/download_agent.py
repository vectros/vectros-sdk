"""
Command-line tool and API for downloading agents from the AIOS Agent Hub.

Provides `download_agent` programmatic function and `main` CLI entry point.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DEFAULT_AGENTHUB_URL = "https://app.aios.foundation"
"""str: Default Agent Hub endpoint URL."""


def download_agent(
    agent_author: str,
    agent_name: str,
    agent_version: Optional[str] = None,
    agenthub_url: str = DEFAULT_AGENTHUB_URL,
    target_dir: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Download an agent package from the AIOS Agent Hub to local storage.

    Args:
        agent_author (str): Author identifier for the agent.
        agent_name (str): Name of the agent to download.
        agent_version (Optional[str], optional): Agent version string. Defaults to None.
        agenthub_url (str, optional): Agent Hub endpoint URL. Defaults to DEFAULT_AGENTHUB_URL.
        target_dir (Optional[str], optional): Destination directory on local machine. Defaults to core agent path.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        Dict[str, Any]: Status dictionary containing success flag, agent name, author, and target path.

    Example:
        >>> from vectros_sdk.commands.download_agent import download_agent
        >>> res = download_agent("demo_author", "demo_agent")
        >>> print(res["success"], res.get("target_dir"))
    """
    base = agenthub_url.rstrip("/")
    params = {"author": agent_author, "name": agent_name}
    if agent_version:
        params["version"] = agent_version

    endpoint = f"{base}/agents/download"

    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
        if not response.ok:
            alt_endpoint = f"{base}/agents/{agent_author}/{agent_name}"
            response = requests.get(alt_endpoint, params={"version": agent_version} if agent_version else None, timeout=timeout)

        if not response.ok:
            error_msg = f"Failed to download agent {agent_author}/{agent_name}: HTTP {response.status_code} - {response.text}"
            return {"success": False, "error": error_msg}

        payload = response.json()

        # Determine target destination directory
        if target_dir:
            dest_dir = Path(target_dir)
        else:
            core_dir = Path(__file__).parent.parent / "agent" / "core"
            dest_dir = core_dir / agent_author / agent_name

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json
        config_data = payload.get("config", {
            "name": agent_name,
            "description": payload.get("description", ""),
            "meta": {
                "author": agent_author,
                "version": agent_version or payload.get("version", "0.0.1"),
                "license": payload.get("license", "MIT"),
            },
            "build": payload.get("build", {
                "entry": "entry.py",
                "module": agent_name.capitalize(),
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
            entry_filename = config_data.get("build", {}).get("entry", "entry.py")
            entry_path = dest_dir / entry_filename
            if not entry_path.exists():
                with open(entry_path, "w", encoding="utf-8") as f:
                    f.write(f"# Auto-generated agent entry for {agent_name}\n")

        return {
            "success": True,
            "agent_name": agent_name,
            "agent_author": agent_author,
            "target_dir": str(dest_dir),
        }

    except Exception as exc:
        return {"success": False, "error": f"Error downloading agent: {exc}"}


def main(args: Optional[List[str]] = None) -> int:
    """
    CLI entry point for `download-agent`.

    Args:
        args (Optional[List[str]], optional): Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        int: Exit status code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="Download an agent from the AIOS Agent Hub.")
    parser.add_argument("--agent_author", required=True, help="Author of the agent")
    parser.add_argument("--agent_name", required=True, help="Name of the agent")
    parser.add_argument("--agent_version", default=None, help="Version of the agent (optional)")
    parser.add_argument("--agenthub_url", default=DEFAULT_AGENTHUB_URL, help=f"Agent Hub URL (default: {DEFAULT_AGENTHUB_URL})")
    parser.add_argument("--target_dir", default=None, help="Local directory to install the agent")

    parsed = parser.parse_args(args)
    res = download_agent(
        agent_author=parsed.agent_author,
        agent_name=parsed.agent_name,
        agent_version=parsed.agent_version,
        agenthub_url=parsed.agenthub_url,
        target_dir=parsed.target_dir,
    )

    if res.get("success"):
        print(f"Successfully downloaded agent '{parsed.agent_name}' by '{parsed.agent_author}' to {res.get('target_dir')}")
        return 0
    else:
        print(f"Failed to download agent: {res.get('error')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
