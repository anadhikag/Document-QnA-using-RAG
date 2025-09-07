# Create demo script and test files

demo_script_content = '''
#!/usr/bin/env python3
"""
Demo script for Document QnA RAG system

This script demonstrates:
1. Document ingestion (PDF/HTML/Markdown)
2. Chunking and indexing
3. Question answering with citations
4. Export functionality

Run with: python demo/demo_script.py
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ingestion.document_processor import DocumentProcessor
from chunking.overlap_chunker import OverlapChunker
from embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from vectorstore.chroma_vectorstore import ChromaVectorStore
from retrieval.top_k_retriever import TopKRetriever
from llm.ollama_provider import OllamaProvider
from utils.config import Config
from utils.helpers import export_to_markdown

def create_sample_documents():
    """Create sample documents for testing"""
    
    sample_dir = Path(__file__).parent / "sample_docs"
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample markdown document
    markdown_content = """# Sample Research Paper

## Abstract

This document demonstrates the Document QnA system's ability to process and answer questions about research papers. The main contribution of this work is the development of a CPU-efficient RAG system.

## Introduction

Retrieval Augmented Generation (RAG) systems combine document retrieval with language model generation to provide factual, grounded responses. This paper presents a local implementation that runs on modest hardware.

## Methodology

Our approach uses the following components:
1. Document ingestion with PyMuPDF for PDFs
2. Semantic chunking with overlap
3. Sentence-Transformers embeddings (all-MiniLM-L6-v2)
4. ChromaDB for vector storage
5. Ollama with phi3 for local generation

### Example Implementation

```python
from document_qna import DocumentProcessor
processor = DocumentProcessor()
documents = processor.ingest_file("paper.pdf")
```

## Results

The system achieves good performance on question answering tasks while maintaining low memory usage. Example questions include:
- "What is the main contribution?"
- "How does the chunking work?"
- "What embedding model is used?"

## Conclusion

This demonstrates a practical approach to building local RAG systems that can run on consumer hardware.
"""
    
    with open(sample_dir / "sample_paper.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    # Create sample HTML document
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>RAG System Guide</title>
</head>
<body>
    <h1>RAG System User Guide</h1>
    
    <h2>Getting Started</h2>
    <p>To get started with the Document QnA system, follow these steps:</p>
    <ol>
        <li>Upload your documents (PDF, HTML, Markdown)</li>
        <li>Click "Index Documents" to process them</li>
        <li>Ask questions about the content</li>
        <li>Export your Q&A session</li>
    </ol>
    
    <h2>Supported Formats</h2>
    <p>The system supports multiple document formats:</p>
    <ul>
        <li><strong>PDF</strong>: Extracted page by page with PyMuPDF</li>
        <li><strong>HTML</strong>: Processed with readability for main content</li>
        <li><strong>Markdown</strong>: Split by headers and sections</li>
        <li><strong>URLs</strong>: Web content fetched and processed</li>
    </ul>
    
    <h2>Technical Details</h2>
    <p>The system uses semantic search to find relevant document sections. 
    Each query is embedded using the same model as the documents, enabling 
    accurate similarity matching.</p>
    
    <h3>Citation Example</h3>
    <p>When you ask "What formats are supported?", the system will reference 
    this page and provide citations like "According to RAG System User Guide, 
    the system supports PDF, HTML, Markdown, and URLs."</p>
</body>
</html>
"""
    
    with open(sample_dir / "user_guide.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Create sample text document
    text_content = """FAQ: Document QnA System

Q: How much memory does the system use?
A: The system is designed to use less than 8GB RAM total, including the OS and other applications. The embeddings model (all-MiniLM-L6-v2) uses about 80MB, and phi3 uses approximately 2-3GB.

Q: Can I use my own documents?
A: Yes! The system supports PDF files, HTML files, Markdown documents, and plain text. You can also provide URLs to process web content.

Q: How accurate are the citations?
A: Citations include the source document name and page/section number when available. For PDFs, page numbers are extracted directly. For other formats, section numbers are used.

Q: Can I run this offline?
A: Absolutely! Once you have downloaded the required models (all-MiniLM-L6-v2 and phi3), the entire system runs locally without internet connectivity.

Q: What if I want to use different models?
A: The system is modular. You can swap the embedding provider (e.g., use OpenAI embeddings) or the LLM provider (e.g., use OpenAI GPT models) by modifying the configuration.

Q: How do I improve answer quality?
A: Try adjusting the chunk size, overlap, or top-k retrieval settings. Also ensure your questions are specific and relate to the content in your documents.
"""
    
    with open(sample_dir / "faq.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    
    print(f"✅ Created sample documents in {sample_dir}")
    return sample_dir

def run_demo():
    """Run the complete demo workflow"""
    
    print("🚀 Starting Document QnA RAG System Demo")
    print("=" * 50)
    
    # Create sample documents
    sample_dir = create_sample_documents()
    
    # Initialize components
    print("\\n📦 Initializing components...")
    config = Config()
    
    try:
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
        print(f"✅ Loaded embedding model: {embeddings.model_name}")
        
        # Initialize vector store
        vectorstore = ChromaVectorStore(
            persist_directory="./demo_chroma_db",
            embeddings=embeddings
        )
        print("✅ Initialized ChromaDB vector store")
        
        # Initialize other components
        processor = DocumentProcessor()
        chunker = OverlapChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        retriever = TopKRetriever(vectorstore, k=config.TOP_K)
        
        # Try to initialize Ollama (graceful fallback if not available)
        try:
            llm = OllamaProvider(model="phi3")
            print("✅ Connected to Ollama with phi3 model")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("📝 Skipping LLM generation (will show retrieval results only)")
            llm = None
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return False
    
    # Phase 1: Document Ingestion
    print("\\n📄 Phase 1: Document Ingestion")
    print("-" * 30)
    
    all_documents = []
    sample_files = [
        sample_dir / "sample_paper.md",
        sample_dir / "user_guide.html", 
        sample_dir / "faq.txt"
    ]
    
    for file_path in sample_files:
        try:
            print(f"Processing {file_path.name}...")
            documents = processor.ingest_file(str(file_path))
            all_documents.extend(documents)
            print(f"  ✅ Extracted {len(documents)} sections/pages")
        except Exception as e:
            print(f"  ❌ Failed to process {file_path.name}: {e}")
    
    print(f"\\n📊 Total documents processed: {len(all_documents)}")
    
    # Phase 2: Chunking and Indexing
    print("\\n🔄 Phase 2: Chunking and Indexing") 
    print("-" * 35)
    
    try:
        # Chunk documents
        print("Chunking documents...")
        all_chunks = []
        for doc in all_documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        chunk_stats = chunker.get_chunk_stats(all_chunks)
        print(f"  ✅ Created {chunk_stats['count']} chunks")
        print(f"  📏 Avg chunk size: {chunk_stats['avg_size']:.0f} chars")
        
        # Add to vector store
        print("Indexing chunks...")
        vectorstore.add_documents(all_chunks)
        print("  ✅ Documents indexed successfully")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False
    
    # Phase 3: Question Answering
    print("\\n💬 Phase 3: Question Answering")
    print("-" * 32)
    
    test_questions = [
        "What is the main contribution of this research?",
        "How much memory does the system use?",
        "What document formats are supported?",
        "Can the system run offline?",
        "What embedding model is used in the methodology?"
    ]
    
    qa_results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\nQuestion {i}: {question}")
        print("-" * (len(question) + 12))
        
        try:
            # Retrieve relevant chunks
            chunks = retriever.query(question)
            
            if chunks:
                print(f"📚 Found {len(chunks)} relevant sources:")
                for j, chunk in enumerate(chunks[:3], 1):  # Show top 3
                    source = chunk.metadata.get('source', 'Unknown')
                    page = chunk.metadata.get('page', chunk.metadata.get('section', 'N/A'))
                    score = chunk.metadata.get('score', 0)
                    preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                    
                    print(f"  {j}. {source} (page/section: {page}) - score: {score:.3f}")
                    print(f"     {preview}")
                
                # Generate answer if LLM is available
                if llm:
                    try:
                        answer = llm.generate_answer(chunks, question)
                        print(f"\\n🤖 Answer: {answer}")
                        qa_results.append((question, answer, chunks))
                    except Exception as e:
                        print(f"❌ Answer generation failed: {e}")
                        qa_results.append((question, f"Error: {e}", chunks))
                else:
                    answer = "LLM not available - showing retrieval results only"
                    qa_results.append((question, answer, chunks))
                    print(f"\\n📝 {answer}")
            else:
                print("❌ No relevant sources found")
                qa_results.append((question, "No relevant information found", []))
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
            qa_results.append((question, f"Query error: {e}", []))
    
    # Phase 4: Export Results
    print("\\n📤 Phase 4: Export Results")
    print("-" * 27)
    
    try:
        markdown_export = export_to_markdown(qa_results)
        
        export_path = Path("demo_qa_results.md")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(markdown_export)
        
        print(f"✅ Results exported to {export_path}")
        print(f"📊 Session contained {len(qa_results)} Q&A pairs")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Summary
    print("\\n🎯 Demo Summary")
    print("-" * 15)
    print(f"📄 Documents processed: {len(all_documents)}")
    print(f"🔤 Chunks created: {len(all_chunks) if 'all_chunks' in locals() else 'N/A'}")
    print(f"❓ Questions answered: {len(qa_results)}")
    print(f"🤖 LLM available: {'Yes' if llm else 'No'}")
    
    collection_stats = vectorstore.get_collection_stats()
    print(f"💾 Vector DB documents: {collection_stats.get('document_count', 'N/A')}")
    
    print("\\n✅ Demo completed successfully!")
    print("\\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Upload your own documents")
    print("3. Explore the interactive Q&A interface")
    
    return True

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
'''

# Create basic test file
test_content = '''
import unittest
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from utils.config import Config
from utils.helpers import clean_text, format_citations
from chunking.overlap_chunker import OverlapChunker
from ingestion.document_processor import Document

class TestDocumentQnA(unittest.TestCase):
    """Basic tests for Document QnA components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.chunker = OverlapChunker(chunk_size=100, overlap=20)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsInstance(self.config.CHUNK_SIZE, int)
        self.assertIsInstance(self.config.EMBEDDING_MODEL, str)
        self.assertGreater(self.config.CHUNK_SIZE, 0)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "  This   has\\nexcessive\\n\\nwhitespace\\t\\t  "
        clean = clean_text(dirty_text)
        self.assertEqual(clean, "This has excessive whitespace")
    
    def test_chunking(self):
        """Test document chunking"""
        document = Document(
            page_content="This is a test document. " * 20,  # Make it long enough to chunk
            metadata={"source": "test.txt", "type": "text"}
        )
        
        chunks = self.chunker.chunk_document(document)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), self.chunker.chunk_size + 50)  # Allow some flexibility
            self.assertIn("chunk_index", chunk.metadata)
    
    def test_citation_formatting(self):
        """Test citation formatting"""
        sources = [
            {"source": "doc1.pdf", "page": 1, "score": 0.85},
            {"source": "doc2.html", "page": "N/A", "score": 0.72}
        ]
        
        citations = format_citations(sources)
        self.assertIn("[1] doc1.pdf", citations)
        self.assertIn("similarity: 0.85", citations)

if __name__ == "__main__":
    unittest.main()
'''

# Write demo and test files
with open(os.path.join("document_qna", "demo", "demo_script.py"), "w", encoding="utf-8") as f:
    f.write(demo_script_content.strip())

with open(os.path.join("document_qna", "tests", "test_basic.py"), "w", encoding="utf-8") as f:
    f.write(test_content.strip())

# Create .env example file
env_example_content = '''# Document QnA Configuration
# Copy this file to .env and adjust values as needed

# Vector database settings
CHROMA_PERSIST_DIR=./chroma_db

# Embedding model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM settings (Ollama local)
LLM_MODEL=ollama:phi3

# Alternative: OpenAI settings (uncomment to use)
# LLM_MODEL=openai:gpt-3.5-turbo
# OPENAI_API_KEY=your-openai-api-key-here

# Chunking settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# Retrieval settings
TOP_K=5
MAX_BATCH_SIZE=32
'''

with open(os.path.join("document_qna", ".env.example"), "w") as f:
    f.write(env_example_content.strip())

print("✅ Created demo script, tests, and environment configuration")
print("✅ Document QnA RAG MVP is complete!")
print()
print("📁 Project structure:")
print("  - app.py (main Streamlit application)")
print("  - src/ (modular source code)")
print("  - demo/ (demo script and sample docs)")
print("  - tests/ (basic unit tests)")
print("  - README.md (comprehensive documentation)")
print("  - requirements.txt (dependencies)")
print("  - .env.example (configuration template)")