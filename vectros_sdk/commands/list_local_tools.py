import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vectros_sdk.tool.core.registry import list_registered_tools


def list_local_tools(tools_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List tools installed locally on the system and registered in the local tool registry.

    Args:
        tools_dir: Optional root directory path to scan for tool config.json files.

    Returns:
        List[Dict[str, Any]]: List of local tool info dictionaries.
    """
    tools = []
    seen_names = set()

    # 1. Registered tools in memory registry
    reg_tools = list_registered_tools()
    for name, tool_cls in reg_tools.items():
        doc = getattr(tool_cls, "__doc__", "") or "Registered Python tool class"
        tools.append({
            "name": name,
            "type": "registered_class",
            "class": getattr(tool_cls, "__name__", str(tool_cls)),
            "description": doc.strip(),
            "source": "TOOL_REGISTRY",
        })
        seen_names.add(name)

    # 2. Filesystem tools scanning
    scan_paths = []
    if tools_dir:
        scan_paths.append(Path(tools_dir))
    else:
        # Default scan in vectros_sdk/tool/core
        core_dir = Path(__file__).parent.parent / "tool" / "core"
        if core_dir.exists():
            scan_paths.append(core_dir)

    for base_dir in scan_paths:
        if not base_dir.exists():
            continue
        for config_path in base_dir.glob("**/config.json"):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                name = cfg.get("name")
                if name and name not in seen_names:
                    meta = cfg.get("meta", {})
                    author = meta.get("author", "local")
                    version = meta.get("version", "0.0.1")
                    desc = cfg.get("description", "")
                    if isinstance(desc, list):
                        desc = " ".join(desc)
                    tools.append({
                        "name": name,
                        "author": author,
                        "version": version,
                        "description": desc,
                        "path": str(config_path.parent),
                        "source": "filesystem",
                    })
                    seen_names.add(name)
            except Exception:
                continue

    return tools


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List tools installed on local system.")
    parser.add_argument(
        "--tools_dir",
        default=None,
        help="Custom directory path to scan for local tools",
    )
    parsed = parser.parse_args(args)

    tools = list_local_tools(tools_dir=parsed.tools_dir)
    if not tools:
        print("No local tools found.")
        return 0

    print(f"=== Locally Available Tools ({len(tools)}) ===")
    for tool in tools:
        name = tool.get("name", "Unknown")
        author = tool.get("author", "N/A")
        version = tool.get("version", "")
        ver_str = f" (v{version})" if version else ""
        desc = tool.get("description", "")
        source = tool.get("source", "local")
        print(f"- {name}{ver_str} [{source}]: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
