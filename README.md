# MCP-Powered-Research-Assistant
A multi-agent research tool built with the Model Context Protocol (MCP). This system orchestrates an LLM to answer user queries by simultaneously retrieving context from local documents (PDFs) and real-time web data, then synthesizing a comparative answer.

## 🚀 Problem Statement
Researchers often need to validate static information (documents/textbooks) against the latest real-time developments (web). Doing this manually requires context switching between reading PDFs and Googling.

This project solves that by:

Ingesting a local PDF (Static Knowledge).

Searching the web via Tavily (Dynamic Knowledge).

Using an LLM to compare and contrast the two sources automatically.

## ⚙️ How It Works (The "Brain Flow")
The system uses Gemini as the central orchestrator, communicating with tools via the Model Context Protocol (JSON-RPC).

Input: User asks a question via the Client/GUI.

Orchestration: Gemini analyzes the request and determines it needs information from both the PDF and the Web.

Parallel Execution (Asyncio): * Tool A (RAG): Extracts text from PDF → Chunks it → Vectorizes it → Retrieves relevant context.

Tool B (Search): Queries Google/Web using the Tavily API.

Synthesis: The LLM receives context from both streams and generates a final answer comparing the two.

## Architecture Diagram
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/313646c1-1998-4b78-8802-fe97f063e6cb" />

## 🛠️ Tech Stack
Core Language: Python 🐍

LLM Orchestrator: Google Gemini 
    ** model="gemini-2.5-flash-lite"

Protocol: Model Context Protocol (MCP)

Web Search: Tavily API

Vector Database & RAG: txtai (w/ FAISS backend)

Concurrency: asyncio (Python standard library)

## 📐 Technical Implementation Details
1.
Vector Embeddings (RAG)
We use a Sentence Transformer model to map text into a "Concept Space."

Model: all-MiniLM-L6-v2 (via txtai)

Parameters: 22 Million

Dimensions: 384-dimensional vector space.

Max Sequence Length: 256 tokens per chunk.

Inductive Bias: The model assumes that nearby points in the vector space share semantic meaning, allowing us to find relevant answers even without exact keyword matches.

2.
Asyncio Concurrency
To ensure low latency, the system utilizes asyncio for true parallelism.

Instead of blocking execution (waiting 3s for PDF + 5s for Web = 8s total), we use await asyncio.sleep() mechanics to run tasks concurrently.

Result: Total time is determined by the slowest single task, not the sum of all tasks.

## 📂 File Structure
Bash
MCP-Research-Assistant/
├── client/
│   └── gui_client.py       # Frontend/Client to send requests
├── server/
│   ├── main.py             # Entry point for MCP Server
│   ├── tools.py            # Definitions for PDF & Tavily tools
│   └── rag_engine.py       # txtai logic (Embeddings & FAISS)
├── data/
│   └── source_docs/        # Place your PDFs here
├── requirements.txt        # Python dependencies
├── .env                    # API Keys (Gemini, Tavily)
└── README.md
## ⚡ Setup & Installation
1. Clone the Repository

Bash
git clone https://github.com/yourusername/mcp-research-assistant.git
cd mcp-research-assistant
2. Create a Virtual Environment

Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies

Bash
pip install -r requirements.txt
4. Configure API Keys
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
5. Run the Server

Bash
python server/main.py
## 🔮 Future Improvements
Implement Recursive Character Splitting for better context retention across chunk boundaries.

Add support for multiple PDF ingestion simultaneously.

Dockerize the application for easier deployment.
