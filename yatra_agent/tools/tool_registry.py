"""Tool definitions and metadata registry for Yatra AI."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolDefinition:
    """Specification of an external tool connected via MCP or local execution."""
    name: str
    description: str
    parameters: Dict[str, Any]
    tool_type: str            # "mcp" | "rag" | "local"
    server_name: str = ""     # MCP server name if applicable
    mcp_tool_name: str = ""   # Tool name on MCP server
    timeout: int = 30         # Timeout in seconds
    is_conditional: bool = False


# Tool definitions will be populated in Phase 3
TOOL_DEFINITIONS = []
