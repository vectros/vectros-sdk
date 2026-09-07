"""
Command-line tool and API for listing tools on the AIOS Tool Hub.

Provides `list_toolhub_tools` programmatic function and `main` CLI entry point.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional
import requests

DEFAULT_TOOLHUB_URL = "https://app.aios.foundation"
"""str: Default Tool Hub endpoint URL."""


def list_toolhub_tools(
    toolhub_url: str = DEFAULT_TOOLHUB_URL,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch and list all tools available on the AIOS Tool Hub.

    Args:
        toolhub_url (str, optional): URL endpoint of the Tool Hub. Defaults to DEFAULT_TOOLHUB_URL.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        List[Dict[str, Any]]: List of tool metadata dictionaries containing name, author, version, and description.

    Example:
        >>> from vectros_sdk.commands.list_toolhub import list_toolhub_tools
        >>> tools = list_toolhub_tools()
        >>> for t in tools:
        ...     print(t["name"])
    """
    url = toolhub_url.rstrip("/")
    endpoint = f"{url}/tools" if not url.endswith("/tools") else url
    try:
        response = requests.get(endpoint, headers={"Accept": "application/json"}, timeout=timeout)
        if response.ok:
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "tools" in data and isinstance(data["tools"], list):
                return data["tools"]
            return [data] if isinstance(data, dict) else []
        print(f"Failed to fetch toolhub tools: HTTP {response.status_code} - {response.text}", file=sys.stderr)
        return []
    except Exception as exc:
        print(f"Error connecting to Tool Hub at {endpoint}: {exc}", file=sys.stderr)
        return []


def main(args: Optional[List[str]] = None) -> int:
    """
    CLI entry point for `list-toolhub-tools`.

    Args:
        args (Optional[List[str]], optional): Command-line argument list. Defaults to sys.argv[1:].

    Returns:
        int: Exit status code (0 for success).
    """
    parser = argparse.ArgumentParser(description="List tools available on AIOS Tool Hub.")
    parser.add_argument(
        "--toolhub_url",
        default=DEFAULT_TOOLHUB_URL,
        help=f"Tool Hub URL (default: {DEFAULT_TOOLHUB_URL})",
    )
    parsed = parser.parse_args(args)

    tools = list_toolhub_tools(toolhub_url=parsed.toolhub_url)
    if not tools:
        print("No tools found on Tool Hub or failed to retrieve.")
        return 0

    print(f"=== Available Tools on Tool Hub ({len(tools)}) ===")
    for tool in tools:
        name = tool.get("name", "Unknown")
        meta = tool.get("meta", {})
        author = meta.get("author", tool.get("author", "Unknown"))
        version = meta.get("version", tool.get("version", "N/A"))
        desc = tool.get("description", "No description")
        if isinstance(desc, list):
            desc = " ".join(desc)
        print(f"- {name} (v{version}) by {author}: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
