import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vectros import Agent, MCPError, MCPServer

SERVER = [sys.executable, str(Path(__file__).parent / "data" / "mcp_server.py")]


@pytest.fixture
def stdio():
    server = MCPServer("fake", SERVER)
    yield server
    server.close()


def test_stdio_server_lists_paginated_tools_with_approval_from_annotations(stdio):
    tools = {t.name: t for t in stdio.tools()}
    assert list(tools) == ["fake__echo", "fake__delete"]
    assert tools["fake__echo"].approval is False      # readOnlyHint
    assert tools["fake__delete"].approval is True     # Not marked read-only.
    assert tools["fake__echo"].parameters["required"] == ["text"]
    assert tools["fake__echo"].invoke({"text": "hi"}) == "hi"
    with pytest.raises(MCPError, match="refused"):
        tools["fake__delete"].invoke({"what": "x"})


def test_tool_filter_and_approval_override():
    with MCPServer("fake", SERVER, tools=["delete"], approval=False) as server:
        assert [(t.name, t.approval) for t in server.tools()] == [("fake__delete", False)]


def test_missing_command_is_reported():
    with pytest.raises(MCPError, match="command not found"):
        MCPServer("x", ["/nonexistent/mcp-server"]).connect()


def test_agent_runs_mcp_tools_as_kernel_client_tools(fake, stdio):
    kernel = fake(replies=[
        {"aios_llm_result": 1, "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "fake__echo", "arguments": "{\"text\": \"pong\"}"}},
            {"id": "c2", "function": {"name": "fake__delete", "arguments": "{}"}}]},
        "done"])
    agent = Agent("m", tools=[stdio], kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results == ["pong", "denied: denied or cancelled by the owner"]
    assert kernel.client_calls[0] == ("fake__echo", {"text": "pong"}, False)
    assert kernel.client_calls[1][2] is True
    names = [spec["function"]["name"] for spec in kernel.llm_calls[0]["tools"]]
    assert names == ["fake__echo", "fake__delete"]


def test_from_config_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN", "s3cret")
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {
        "gh": {"url": "https://example.invalid/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},
        "off": {"command": "x", "disabled": True},
        "local": {"command": "srv --flag", "args": ["a"], "side_effect": ["rm"]}}}))
    servers = MCPServer.from_config(path)
    assert [s.name for s in servers] == ["gh", "local"]
    assert servers[0].headers["Authorization"] == "Bearer s3cret"
    assert servers[1].command == ["srv", "--flag", "a"] and servers[1].approval == {"rm"}


def test_streamable_http_server_with_sse_replies():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((message.get("method"), self.headers.get("Mcp-Session-Id")))
            if "id" not in message:
                self.send_response(202)
                self.end_headers()
                return
            if message["method"] == "initialize":
                result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            elif message["method"] == "tools/list":
                result = {"tools": [{"name": "time.now", "description": "Time.",
                                     "annotations": {"readOnlyHint": True}}]}
            else:
                result = {"content": [], "structuredContent": {"time": "noon"}}
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result})}\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Mcp-Session-Id", "sess-1")
            self.end_headers()
            self.wfile.write(body.encode())

        def do_DELETE(self):
            seen.append(("DELETE", self.headers.get("Mcp-Session-Id")))
            self.send_response(200)
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        server = MCPServer("clock", url=f"http://127.0.0.1:{httpd.server_port}/mcp")
        (item,) = server.tools()
        assert item.name == "clock__time_now"  # Dots are not valid in function names.
        assert item.invoke({}) == '{"time": "noon"}'
        server.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert seen[0] == ("initialize", None)
    assert ("tools/call", "sess-1") in seen and seen[-1] == ("DELETE", "sess-1")
