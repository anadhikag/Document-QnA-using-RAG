import os
import sys

# Create the complete project structure for Document QnA RAG MVP
project_structure = {
    "README.md": """# Document QnA (RAG) — Local Demo for i5/8GB

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
""",
    
    "requirements.txt": """streamlit>=1.28.0
chromadb>=0.4.15
sentence-transformers>=2.2.2
pymupdf>=1.23.0
beautifulsoup4>=4.12.0
readability-lxml>=0.8.1
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.66.0
ollama>=0.1.7
pandas>=2.0.0
numpy>=1.24.0
""",
    
    "app.py": """
import streamlit as st
import os
import sys
from typing import List, Dict, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.document_processor import DocumentProcessor
from chunking.overlap_chunker import OverlapChunker
from embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from vectorstore.chroma_vectorstore import ChromaVectorStore
from retrieval.top_k_retriever import TopKRetriever
from llm.ollama_provider import OllamaProvider
from utils.config import Config
from utils.helpers import format_citations, export_to_markdown

# Page configuration
st.set_page_config(
    page_title="Document QnA (RAG)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
config = Config()

# Initialize components
@st.cache_resource
def init_components():
    \"\"\"Initialize and cache the RAG components\"\"\"
    try:
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        # Initialize vector store
        vectorstore = ChromaVectorStore(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embeddings=embeddings
        )
        
        # Initialize other components
        processor = DocumentProcessor()
        chunker = OverlapChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        retriever = TopKRetriever(vectorstore, k=config.TOP_K)
        llm = OllamaProvider(model=config.LLM_MODEL.split(':')[1] if ':' in config.LLM_MODEL else config.LLM_MODEL)
        
        return processor, chunker, embeddings, vectorstore, retriever, llm
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

# Initialize session state
def init_session_state():
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'indexed_count' not in st.session_state:
        st.session_state.indexed_count = 0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ""

def main():
    init_session_state()
    
    # Load components
    processor, chunker, embeddings, vectorstore, retriever, llm = init_components()
    
    # Header
    st.title("📚 Document QnA (RAG)")
    st.markdown("*Local CPU-based Question & Answer system with document citations*")
    
    # Sidebar for document upload and indexing
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'html', 'md', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, HTML, Markdown, Text"
        )
        
        # URL input
        url_input = st.text_input(
            "Or enter URL",
            placeholder="https://example.com/document.html",
            help="Enter a URL to index web content"
        )
        
        # Process uploaded files
        if uploaded_files:
            st.session_state.documents = []
            for file in uploaded_files:
                try:
                    # Save temporarily and process
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    docs = processor.ingest_file(temp_path)
                    st.session_state.documents.extend(docs)
                    os.remove(temp_path)  # Clean up
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
        
        # Process URL
        if url_input and st.button("Load URL"):
            try:
                with st.spinner("Loading URL content..."):
                    docs = processor.ingest_url(url_input)
                    st.session_state.documents.extend(docs)
                    st.success(f"Loaded content from URL")
            except Exception as e:
                st.error(f"Error loading URL: {str(e)}")
        
        # Display document count
        if st.session_state.documents:
            st.success(f"📄 {len(st.session_state.documents)} documents loaded")
            
            # Index documents button
            if st.button("🔄 Index Documents", type="primary"):
                try:
                    with st.spinner("Indexing documents..."):
                        # Chunk documents
                        all_chunks = []
                        for doc in st.session_state.documents:
                            chunks = chunker.chunk_document(doc)
                            all_chunks.extend(chunks)
                        
                        # Add to vector store
                        vectorstore.add_documents(all_chunks)
                        st.session_state.indexed_count = len(all_chunks)
                        st.session_state.processing_status = f"✅ Indexed {len(all_chunks)} chunks from {len(st.session_state.documents)} documents"
                        st.success(st.session_state.processing_status)
                        
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)}")
        
        # Show indexing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        st.markdown("---")
        
        # Export chat history
        if st.session_state.chat_history:
            st.header("📤 Export")
            if st.button("Export Q&A as Markdown"):
                markdown_content = export_to_markdown(st.session_state.chat_history)
                st.download_button(
                    label="Download Markdown",
                    data=markdown_content,
                    file_name=f"qna_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    # Main chat interface
    if st.session_state.indexed_count > 0:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                st.write(answer)
                
                # Show sources
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"**Source {j}**: {source.get('source', 'Unknown')}")
                            st.markdown(f"*Page/Section*: {source.get('page', 'N/A')}")
                            st.markdown(f"*Similarity*: {source.get('score', 0):.3f}")
                            st.markdown(f"*Content*: {source.get('content', '')[:200]}...")
                            if j < len(sources):
                                st.markdown("---")
        
        # Question input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching documents and generating answer..."):
                        # Retrieve relevant chunks
                        retrieved_chunks = retriever.query(user_question)
                        
                        if not retrieved_chunks:
                            st.write("I couldn't find relevant information in the indexed documents.")
                            answer = "No relevant information found."
                            sources = []
                        else:
                            # Generate answer
                            answer = llm.generate_answer(retrieved_chunks, user_question)
                            
                            # Format sources
                            sources = []
                            for chunk in retrieved_chunks:
                                sources.append({
                                    'source': chunk.metadata.get('source', 'Unknown'),
                                    'page': chunk.metadata.get('page', 'N/A'),
                                    'score': chunk.metadata.get('score', 0),
                                    'content': chunk.page_content
                                })
                            
                            st.write(answer)
                            
                            # Show sources
                            with st.expander(f"📚 Sources ({len(sources)})"):
                                for j, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {j}**: {source['source']}")
                                    st.markdown(f"*Page/Section*: {source['page']}")
                                    st.markdown(f"*Similarity*: {source['score']:.3f}")
                                    st.markdown(f"*Content*: {source['content'][:200]}...")
                                    if j < len(sources):
                                        st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, answer, sources))
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.write("I encountered an error while processing your question. Please try again.")
    
    else:
        # Welcome screen
        st.header("🚀 Welcome to Document QnA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How to get started:
            
            1. **Upload documents** in the sidebar (PDF, HTML, Markdown)
            2. **Click "Index Documents"** to process and store them
            3. **Ask questions** about your documents
            4. **Get answers with citations** showing sources
            5. **Export your Q&A session** as Markdown
            """)
        
        with col2:
            st.markdown("""
            ### Features:
            
            - 🏠 **Runs locally** on your CPU (no cloud required)
            - 📄 **Multi-format support** (PDF, HTML, MD, URLs)  
            - 🧠 **Semantic search** with embeddings
            - 💾 **Persistent storage** (indexes saved to disk)
            - 📚 **Source citations** with page references
            - 💬 **Chat history** and export functionality
            """)
        
        # Sample usage
        st.markdown("---")
        st.header("📋 Sample Usage")
        
        st.markdown("""
        **Example workflow:**
        1. Upload `ExamplePaper.pdf` → Click **Index Documents** → ✅ Success message
        2. Ask: *"What is the main contribution?"* → Get answer with citations *(page 2, paragraph 1)*
        3. Follow-up: *"Give one example mentioned."* → Get excerpt with cited page
        4. Click **Export Q&A as Markdown** → Download complete session
        """)
        
        # System info
        with st.expander("⚙️ System Information"):
            st.markdown(f"""
            - **Embedding Model**: {config.EMBEDDING_MODEL}
            - **LLM Model**: {config.LLM_MODEL}
            - **Chunk Size**: {config.CHUNK_SIZE} characters
            - **Chunk Overlap**: {config.CHUNK_OVERLAP} characters
            - **Retrieval Top-K**: {config.TOP_K}
            - **Vector DB**: ChromaDB (persistent)
            - **Storage**: {config.CHROMA_PERSIST_DIR}
            """)

if __name__ == "__main__":
    main()
""",
}

# Create project directory and files
project_dir = "document_qna"
if not os.path.exists(project_dir):
    os.makedirs(project_dir)

# Write main files
for filename, content in project_structure.items():
    filepath = os.path.join(project_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip())

print(f"✅ Created main project files in {project_dir}/")
print("📁 Files created:")
for filename in project_structure.keys():
    print(f"  - {filename}")