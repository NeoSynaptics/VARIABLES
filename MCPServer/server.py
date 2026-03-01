"""
MCPServer — Model Context Protocol server over stdio (JSON-RPC 2.0).

Minimal MCP server that exposes tools to LLM agents via the standard MCP protocol.
Communicates over stdin/stdout using JSON-RPC 2.0 message format.

Usage:
    server = MCPServer("my-server")
    server.tool("greet", "Greet a user", {"name": {"type": "string"}})(greet_handler)
    server.run()  # blocks, reads from stdin
"""
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    handler: Callable = lambda: ""


class MCPServer:
    """JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self, name: str = "mcp-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: dict[str, Tool] = {}
        self.resources: dict[str, Resource] = {}

    def tool(self, name: str, description: str, parameters: dict[str, Any] = None):
        """Decorator to register a tool."""
        def decorator(fn: Callable):
            self.tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters or {},
                handler=fn,
            )
            return fn
        return decorator

    def resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        """Decorator to register a resource."""
        def decorator(fn: Callable):
            self.resources[uri] = Resource(
                uri=uri, name=name, description=description,
                mime_type=mime_type, handler=fn,
            )
            return fn
        return decorator

    def _send(self, msg: dict):
        """Send a JSON-RPC message to stdout."""
        raw = json.dumps(msg)
        sys.stdout.write(f"Content-Length: {len(raw)}

{raw}")
        sys.stdout.flush()

    def _respond(self, id: Any, result: Any):
        self._send({"jsonrpc": "2.0", "id": id, "result": result})

    def _error(self, id: Any, code: int, message: str):
        self._send({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})

    def _handle(self, msg: dict):
        method = msg.get("method", "")
        id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            self._respond(id, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
            })
        elif method == "tools/list":
            tools_list = [
                {"name": t.name, "description": t.description,
                 "inputSchema": {"type": "object", "properties": t.parameters}}
                for t in self.tools.values()
            ]
            self._respond(id, {"tools": tools_list})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            if tool_name not in self.tools:
                self._error(id, -32601, f"Unknown tool: {tool_name}")
                return
            try:
                result = self.tools[tool_name].handler(**arguments)
                content = [{"type": "text", "text": str(result)}]
                self._respond(id, {"content": content})
            except Exception as e:
                self._respond(id, {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
        elif method == "resources/list":
            res_list = [
                {"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                for r in self.resources.values()
            ]
            self._respond(id, {"resources": res_list})
        elif method == "resources/read":
            uri = params.get("uri", "")
            if uri not in self.resources:
                self._error(id, -32601, f"Unknown resource: {uri}")
                return
            r = self.resources[uri]
            content = r.handler()
            self._respond(id, {"contents": [{"uri": uri, "mimeType": r.mime_type, "text": str(content)}]})
        elif method == "notifications/initialized":
            pass  # client acknowledgement, no response needed
        else:
            if id is not None:
                self._error(id, -32601, f"Method not found: {method}")

    def run(self):
        """Main loop: read JSON-RPC messages from stdin."""
        buf = ""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                buf += line
                # Try to parse Content-Length header + body
                if "Content-Length:" in buf:
                    header_end = buf.find("

")
                    if header_end == -1:
                        header_end = buf.find("

")
                    if header_end >= 0:
                        header = buf[:header_end]
                        length = int(header.split("Content-Length:")[-1].strip().split()[0])
                        body_start = header_end + 4 if "

" in buf[:header_end + 5] else header_end + 2
                        if len(buf) >= body_start + length:
                            body = buf[body_start:body_start + length]
                            buf = buf[body_start + length:]
                            msg = json.loads(body)
                            self._handle(msg)
                else:
                    # Try raw JSON (for simple clients)
                    try:
                        msg = json.loads(buf.strip())
                        buf = ""
                        self._handle(msg)
                    except json.JSONDecodeError:
                        pass
            except KeyboardInterrupt:
                break
            except Exception:
                pass
