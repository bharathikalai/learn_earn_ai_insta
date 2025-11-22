# mcp_server.py
import sys
from fastmcp import FastMCP

mcp = FastMCP("demo-mcp-server")

@mcp.tool()
async def echo(message: str) -> str:
    """Echo back the message."""
    return message

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

@mcp.tool()
def read_greeting() -> str:
    """Read and return the contents of greeting.txt file."""
    try:
        with open("greeting.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: greeting.txt not found"
@mcp.tool()
def read_students() -> str:
    """Read and return the contents of student.txt file."""
    try:
        with open("student_roster.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: student_roster.txt not found"

@mcp.prompt()
async def greeting_prompt(name: str) -> str:
    """A simple greeting prompt."""
    return f"Greet {name} kindly  lusu payalea."

@mcp.resource("file://./greeting.txt")
def greeting_file() -> str:
    """Read and return the contents of greeting.txt."""
    with open("greeting.txt", "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    print("Starting MCP SSE server on port 8000...", file=sys.stderr)
    # Try SSE transport instead
    mcp.run(transport="http", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

