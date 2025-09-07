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
        dirty_text = "  This   has\nexcessive\n\nwhitespace\t\t  "
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