import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional
from mcp.server.mcpserver import MCPServer

# setup logger để in ra stderr (in ra vào file log để kiểm tra lỗi)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_server")

mcp = MCPServer("LessonPlanningAssistantServer")

@mcp.tool()
def abc():
    pass

if __name__ == "__main__":
    mcp.run(transport="stdio")