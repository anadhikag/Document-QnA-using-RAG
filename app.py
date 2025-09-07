import streamlit as st
import os
import sys
from datetime import datetime

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ingestion.document_processor import DocumentProcessor
from chunking.overlap_chunker import OverlapChunker, Chunk
from embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from vectorstore.chroma_vectorstore import ChromaVectorStore
from retrieval.top_k_retriever import TopKRetriever
from utils.config import Config
from utils.helpers import export_to_markdown
from llm.ollama_provider import OllamaProvider

st.set_page_config(
    page_title="Document QnA (Ollama)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

config = Config()

@st.cache_resource
def init_components():
    """Initialize and cache the RAG components."""
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = ChromaVectorStore(persist_directory=config.CHROMA_PERSIST_DIR, embeddings=embeddings)
    processor = DocumentProcessor()
    chunker = OverlapChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    retriever = TopKRetriever(vectorstore, k=config.TOP_K)
    model_name = config.LLM_MODEL.lower().split(":")[1]
    llm = OllamaProvider(model=model_name)
    st.info(f"Connected to local LLM (Ollama): {model_name}")
    return processor, chunker, vectorstore, retriever, llm

def init_session_state():
    if "documents" not in st.session_state: st.session_state.documents = []
    if "indexed_count" not in st.session_state: st.session_state.indexed_count = 0
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "processing_status" not in st.session_state: st.session_state.processing_status = ""

def main():
    st.title("📚 Document QnA with Ollama")
    processor, chunker, vectorstore, retriever, llm = init_components()
    init_session_state()

    with st.sidebar:
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "html", "md", "txt"], accept_multiple_files=True)
        url_input = st.text_input("Or enter a URL to scrape")

        if (uploaded_files or url_input) and st.button("Load Documents", type="secondary"):
            st.session_state.documents = []
            with st.spinner("Processing documents..."):
                if uploaded_files:
                    for file in uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f: f.write(file.getvalue())
                        try:
                            # --- KEY CHANGE: Pass the original filename ---
                            docs = processor.ingest_file(temp_path, original_filename=file.name)
                            st.session_state.documents.extend(docs)
                        finally:
                            if os.path.exists(temp_path): os.remove(temp_path)
                if url_input:
                    docs = processor.ingest_url(url_input)
                    st.session_state.documents.extend(docs)
            st.success(f"Loaded {len(st.session_state.documents)} document sections.")
        
        if st.session_state.documents:
            # --- KEY CHANGE: Updated button text and logic ---
            if st.button("🔄 Index New Documents", type="primary"):
                with st.spinner("Clearing old data and indexing... This may take a moment."):
                    # --- "CLEAR THE BOOKSHELF" ---
                    vectorstore.clear_collection()
                    
                    all_chunks = chunker.chunk_documents(st.session_state.documents)
                    vectorstore.add_documents(all_chunks)
                    st.session_state.indexed_count = len(all_chunks)
                    st.session_state.processing_status = f"✅ Indexed {len(all_chunks)} new chunks."
                    st.success(st.session_state.processing_status)
                    st.session_state.documents = []
                    # Also clear chat history for a fresh start
                    st.session_state.chat_history = []

    if st.session_state.indexed_count > 0:
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chunks = retriever.query(prompt)
                    if not chunks:
                        response = "I could not find relevant information to answer that."
                    else:
                        response = llm.generate_answer(chunks, prompt)
                        with st.expander("View Sources"):
                            for chunk in chunks:
                                st.info(f"Source: {chunk.metadata.get('source', 'N/A')}, Score: {chunk.metadata.get('score', 0):.2f}\n\n> {chunk.page_content[:200]}...")
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Welcome! Please upload and index documents to begin.")

if __name__ == "__main__":
    main()

