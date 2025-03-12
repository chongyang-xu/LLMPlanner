from typing import Dict, List, Literal, Optional


class MCPServerSettings:
    """
    Represents the configuration for an individual server.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        transport: Literal["stdio", "sse"] = "stdio",
        command: str | None = None,
        args: List[str] | None = None,
        env: Dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.transport = transport
        self.command = command
        self.args = args
        self.env = env

    name: str | None = None
    """The name of the server."""

    description: str | None = None
    """The description of the server."""

    transport: Literal["stdio", "sse"] = "stdio"
    """The transport mechanism."""

    command: str | None = None
    """The command to execute the server (e.g. npx)."""

    args: List[str] | None = None
    """The arguments for the server command."""

    env: Dict[str, str] | None = None
    """Environment variables to pass to the server process."""


class MCPSettings:
    """Configuration for all MCP servers."""

    def __init__(self,
                 servers: Dict[str, MCPServerSettings] | None = None) -> None:
        self.servers = servers if servers is not None else {}

    servers: Dict[str, MCPServerSettings] = {}


global mcp_settings
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
echo_server_path = os.path.join(current_dir, '..', '..', 'service',
                                'echo_server.py')

mcp_settings = MCPSettings(
    servers={
        "fetch":
            MCPServerSettings(
                command="uvx",
                args=["mcp-server-fetch"],
            ),
        "filesystem":
            MCPServerSettings(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem"],
            ),
        "time":
            MCPServerSettings(
                command="uvx",
                args=["mcp-server-time"],
            ),
        "echo":
            MCPServerSettings(
                command="python",
                args=[echo_server_path],
            ),
    })
