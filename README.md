# Document QnA (RAG) — Local Demo for i5/8GB

A local Document Question & Answer system using Retrieval Augmented Generation (RAG) that runs on CPU with minimal memory requirements.

## Features

- **Multi-format document support**: PDF, HTML, Markdown, and URLs
- **Persistent vector database**: ChromaDB with local storage
- **CPU-optimized embeddings**: Sentence-Transformers all-MiniLM-L6-v2
- **Local LLM**: Ollama with phi3 model (default)
- **Modular architecture**: Easy to swap embeddings and LLM providers
- **Streamlit UI**: Upload files, index documents, ask questions, export results
- **Citation support**: Shows sources with page/paragraph references
- **Session management**: Chat history and document persistence

## Hardware Requirements

- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB (system will use ≤6GB)
- **Storage**: 2GB free space for models and indexes
- **OS**: Windows, macOS, or Linux

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install Ollama and download phi3**:
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Then download phi3 model
ollama pull phi3
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Demo workflow**:
   - Upload a document (PDF/HTML/MD) or enter a URL
   - Click "Index Documents" 
   - Ask questions about your documents
   - Export Q&A session as Markdown

## Configuration

Environment variables (optional):
```bash
CHROMA_PERSIST_DIR=./chroma_db  # Vector database location
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Embedding model name
LLM_MODEL=ollama:phi3  # LLM provider and model
CHUNK_SIZE=1000  # Characters per chunk
CHUNK_OVERLAP=100  # Overlap between chunks
TOP_K=5  # Number of chunks to retrieve
```

## Project Structure

```
document_qna/
├── app.py                 # Main Streamlit application
├── src/
│   ├── ingestion/         # Document parsing and loading
│   ├── chunking/          # Text chunking with overlap
│   ├── embeddings/        # Embedding providers (Sentence-Transformers)
│   ├── vectorstore/       # ChromaDB integration
│   ├── retrieval/         # Top-k retrieval with similarity
│   ├── llm/               # LLM providers (Ollama, OpenAI)
│   └── utils/             # Common utilities
├── tests/
├── demo/
│   ├── sample_docs/       # Sample documents for testing
│   └── demo_script.py     # Automated demo script
└── requirements.txt
```

## Modular Provider System

### Swap Embedding Provider
```python
# Default: Sentence-Transformers
from src.embeddings import SentenceTransformerEmbeddings

# Alternative: OpenAI embeddings  
from src.embeddings import OpenAIEmbeddings
embedder = OpenAIEmbeddings(api_key="your-key")
```

### Swap LLM Provider
```python
# Default: Ollama local
from src.llm import OllamaProvider

# Alternative: OpenAI
from src.llm import OpenAIProvider
llm = OpenAIProvider(api_key="your-key", model="gpt-3.5-turbo")
```

## Usage Examples

### Basic Document Processing
```python
from src.ingestion import DocumentProcessor
from src.chunking import OverlapChunker
from src.vectorstore import ChromaVectorStore

# Process document
processor = DocumentProcessor()
documents = processor.ingest_file("document.pdf")

# Create chunks
chunker = OverlapChunker(chunk_size=1000, overlap=100)
chunks = chunker.chunk_documents(documents)

# Store in vector database
vectorstore = ChromaVectorStore(persist_directory="./chroma_db")
vectorstore.add_documents(chunks)
```

### Query Documents
```python
from src.retrieval import TopKRetriever
from src.llm import OllamaProvider

# Retrieve relevant chunks
retriever = TopKRetriever(vectorstore, k=5)
chunks = retriever.query("What is the main contribution?")

# Generate answer
llm = OllamaProvider(model="phi3")
answer = llm.generate_answer(chunks, "What is the main contribution?")
```

## Testing

Run the demo script to test all functionality:
```bash
python demo/demo_script.py
```

This will:
1. Index sample documents (PDF, HTML, Markdown)
2. Ask 5 test questions with expected citations
3. Export results to Markdown
4. Verify all components work correctly

## Troubleshooting

**Out of Memory**: Reduce `CHUNK_SIZE` or `TOP_K` values
**Slow Performance**: Ensure phi3 model is downloaded locally
**Import Errors**: Check all requirements are installed
**ChromaDB Issues**: Delete `./chroma_db` folder to reset database

## Architecture

The system follows a modular RAG pipeline:

1. **Document Ingestion**: Parse PDFs/HTML/MD/URLs into structured text
2. **Chunking**: Split documents into overlapping semantic chunks  
3. **Embedding**: Convert chunks to dense vectors using all-MiniLM-L6-v2
4. **Vector Storage**: Store embeddings in persistent ChromaDB
5. **Retrieval**: Find top-k similar chunks for user queries
6. **Generation**: Use Ollama+phi3 to generate grounded answers with citations

## License

MIT License - see LICENSE file for details