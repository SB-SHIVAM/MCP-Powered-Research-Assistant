import os
from txtai.embeddings import Embeddings
from pypdf import PdfReader

# Configuration
INDEX_PATH = os.path.join(os.getcwd(), "data", "txtai_index")

# Initialize txtai Embeddings (using a small, fast model)
# We declare it global so we keep it in memory while the server runs
embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

def extract_text_chunks(pdf_path: str):
    """Reads PDF and splits it into manageable chunks."""
    print(f" Reading PDF: {pdf_path}...")
    reader = PdfReader(pdf_path)
    data = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Simple chunking by paragraph
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if len(para) > 50:  # Filter out tiny noises
                    # Format: (id, text, metadata)
                    # We store the page number in the text so the AI sees it
                    data.append((i, f"[Page {i+1}] {para}", None))
    return data

async def pdf_agent(query: str, pdf_path: str) -> str:
    """
    Searches the PDF using txtai vector embeddings.
    Auto-loads the index if it exists, or builds it if missing.
    """
    global embeddings

    # 1. Check if Index exists on disk
    if os.path.exists(INDEX_PATH):
        if not embeddings.count(): # If not loaded in memory yet
            print("Loading existing txtai index...")
            embeddings.load(INDEX_PATH)
    else:
        # 2. If no index, build it from scratch
        print("No index found. Building new txtai index...")
        data = extract_text_chunks(pdf_path)
        
        if not data:
            return "Error: Could not extract text from PDF."

        print(f"Indexing {len(data)} text chunks...")
        embeddings.index(data)
        
        print(f" Saving index to {INDEX_PATH}...")
        embeddings.save(INDEX_PATH)

    # 3. Perform Vector Search
    print(f"🔍 Vector searching for: '{query}'")
    results = embeddings.search(query, limit=3)

    # 4. Format Results
    output = "Found in Study Notes (via txtai):\n"
    for result in results:
        # result is typically a dict or tuple depending on txtai version
        # We usually get {'text': '...', 'score': ...} or just the text if simple
        text = result['text'] if isinstance(result, dict) else result[0]
        output += f"\n---\n{text}\n"
        
    return output