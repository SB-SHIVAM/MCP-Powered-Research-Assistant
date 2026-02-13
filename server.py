import sys
import io
import os
from dotenv import load_dotenv

# 1. LOAD DOTENV FIRST (Before importing agents that use API keys)
load_dotenv()

# 2. FIX WINDOWS ENCODING (Ensures special characters don't crash the pipe)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

import asyncio
from mcp.server.fastmcp import FastMCP

# 3. IMPORT AGENTS
from app.pdf_agent import pdf_agent
from app.tavily_agent import tavily_agent

# Configuration
PDF_PATH = os.path.join(os.getcwd(), "data", "Langchain_STUDY.pdf")

# Initialize MCP Server
mcp = FastMCP("Research Assistant Server")

@mcp.tool()
async def search_study_notes(query: str) -> str:
    """
    Search the local PDF for technical definitions and course content.
    """
    # Note: pdf_agent handles txtai indexing/loading internally
    return await pdf_agent(query, PDF_PATH)

@mcp.tool()
async def search_internet(query: str) -> str:
    """
    Search the web for up-to-date information and modern libraries.
    """
    return await tavily_agent(query)

if __name__ == "__main__":
    # FastMCP handles the run loop over stdio
    mcp.run()