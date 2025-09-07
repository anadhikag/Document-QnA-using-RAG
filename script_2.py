# Create the main Streamlit app.py file
app_py_content = '''
import streamlit as st
import os
import sys
from typing import List, Dict, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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
    """Initialize and cache the RAG components"""
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
        llm = OllamaProvider(model=config.LLM_MODEL.split(":")[1] if ":" in config.LLM_MODEL else config.LLM_MODEL)
        
        return processor, chunker, embeddings, vectorstore, retriever, llm
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

# Initialize session state
def init_session_state():
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "indexed_count" not in st.session_state:
        st.session_state.indexed_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_status" not in st.session_state:
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
            type=["pdf", "html", "md", "txt"],
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
                            st.write("I could not find relevant information in the indexed documents.")
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
            
            - **Runs locally** on your CPU (no cloud required)
            - **Multi-format support** (PDF, HTML, MD, URLs)  
            - **Semantic search** with embeddings
            - **Persistent storage** (indexes saved to disk)
            - **Source citations** with page references
            - **Chat history** and export functionality
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
'''

# Write app.py
with open(os.path.join("document_qna", "app.py"), "w", encoding="utf-8") as f:
    f.write(app_py_content.strip())

print("✅ Created app.py file")