"""A tiny stdio MCP server for tests: tools echo (read-only) and delete."""

import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo text back.", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "delete", "description": "Delete a thing.",
     "inputSchema": {"type": "object", "properties": {"what": {"type": "string"}}}},
]

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method, params = message["method"], message.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        if not params.get("cursor"):
            result = {"tools": TOOLS[:1], "nextCursor": "2"}
        else:
            result = {"tools": TOOLS[1:]}
    elif method == "tools/call":
        args = params.get("arguments") or {}
        if params["name"] == "echo":
            result = {"content": [{"type": "text", "text": args["text"]}]}
        else:
            result = {"content": [{"type": "text", "text": "refused"}], "isError": True}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                          "error": {"code": -32601, "message": "no"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
