# Week 7: Model Context Protocol (MCP) & External Tool Integration

## 1. General Information & Learning Objectives (A7.1 to A7.6)
- **A7.1**: Explain the Host / Client / Server architecture, the 3 Primitives (Resources, Prompts, Tools) and the 2 Transports (stdio, SSE).
- **A7.2**: Add and configure community MCP Servers from a config file without rewriting the application source code.
- **A7.3**: Inspect and Audit the Tool inventory, the JSON schema and the security boundary (Trust boundary).
- **A7.4**: Write a Python MCP Client that connects over `stdio`, discovers tools with `list_tools()` and calls them with `call_tool()`.
- **A7.5**: Write a standard MCP Server yourself, wrap a VectorStore as an MCP tool and test it with a JSON-RPC probe.
- **A7.6**: Build a dynamic ReAct Agent whose tool set comes entirely from `list_tools()` without hardcoding it in the source code.

## 2. Foundational Theory & System Architecture
### 2.1. Why is the Model Context Protocol (MCP) inevitable?
- **The $N \times M$ problem**: Before MCP, with $N$ AI models/Hosts and $M$ tools/data sources, developers had to write $N \times M$ separate adapters.
- **The MCP solution**: Standardize a single unified protocol (based on JSON-RPC 2.0). Every data source becomes an MCP Server; every AI application becomes an MCP Client. The number of connections drops to $N + M$.

### 2.2. Core components & concepts of MCP
- **Three Roles**:
  - **Host**: The high-level orchestrating application (Claude Desktop, an IDE, or a Python ReAct Agent).
  - **Client**: The module that establishes a $1:1$ connection with one specific Server.
  - **Server**: An independent application that exposes data and features through the standard protocol.
- **Three core Primitives**:
  - `Tools`: Action functions the LLM can call to execute (creating a Side Effect or performing a computation).
  - `Resources`: Read-only context data (files, logs, database schemas).
  - `Prompts`: Prebuilt prompt templates supplied by the server.
- **Two Transports**:
  - `stdio`: Communication through the standard input/output streams of a local process (Local process stdin/stdout).
  - `sse` (Server-Sent Events): Communication over HTTP for remote services.

### 2.3. Trust boundary & the mandatory Stdio Transport rule
- **Trust Boundary**: The Server is outside the direct control of the LLM. The Client must audit the schema and the arguments before executing.
- **The Stdio golden rule**: Absolutely **DO NOT USE `print()`** carelessly in an MCP Server running over `stdio`, because it corrupts the JSON-RPC packet structure on the stdout stream. All logs must go to `sys.stderr`.

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 7.1 — Building an MCP Server for the course (src/mcp_service/server.py)
```python
from mcp.server import MCPServer
import sys

mcp = MCPServer("aipr-study-assistant")

@mcp.tool()
def register_study_task(task_id: str, week: int, title: str, due_date: str) -> str:
    """Register a study task that can be verified (Write Side Effect)."""
    # Write the log through stderr to protect stdout for JSON-RPC
    print(f"[MCP-LOG] Registering task {task_id} for week {week}", file=sys.stderr)
    return f"Task '{title}' has been recorded in persistent storage."

@mcp.tool()
def get_study_tasks(week: int = 0) -> str:
    """Retrieve the list of registered tasks."""
    return "Current study tasks: [HW_W04, HW_W07]"
```

### Exercise 7.2 — Python MCP Client connecting over Stdio (ex2_1_mcp_client.py)
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    server_params = StdioServerParameters(command="python", args=["src/mcp_service/server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Dynamic Tool Discovery
            tools = await session.list_tools()
            print(f"Discovered {len(tools.tools)} tools from the server.")
            # Execute a tool
            res = await session.call_tool("register_study_task", {"task_id": "T01", "week": 4, "title": "Lab BM25", "due_date": "2026-10-25"})
            print("Result:", res.content[0].text)
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **JSON-RPC stream breakage from print pitfall**: Any library that accidentally prints to stdout (for example a model-loading warning) crashes the MCP stdio connection.
- **No Side Effect pitfall**: A project that uses MCP but only writes read-only tools will fail the project criteria; there must be a Write tool that saves to disk or a database.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Build an independent JSON-RPC probe test to verify the stability of the MCP Server before letting the Agent connect to it.
