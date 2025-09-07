import os
import re
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pymupdf
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

# Use the correct absolute import
from utils.helpers import clean_text

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    # --- KEY CHANGE: Added 'original_filename' parameter ---
    def ingest_file(self, file_path: str, original_filename: Optional[str] = None) -> List[Document]:
        """Process a file and return a list of documents."""
        source_name = original_filename or os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            return self._process_pdf(file_path, source_name)
        elif file_ext in ['.html', '.htm']:
            return self._process_html_file(file_path, source_name)
        elif file_ext in ['.md', '.markdown']:
            return self._process_markdown_file(file_path, source_name)
        elif file_ext == '.txt':
            return self._process_text_file(file_path, source_name)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def ingest_url(self, url: str) -> List[Document]:
        """Process content from a URL."""
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return self._process_html_content(response.text, url)

    def _process_pdf(self, file_path: str, source_name: str) -> List[Document]:
        documents = []
        with pymupdf.open(file_path) as pdf_doc:
            for page_num, page in enumerate(pdf_doc):
                text = page.get_text("text")
                if text.strip():
                    documents.append(Document(
                        page_content=clean_text(text),
                        metadata={'source': source_name, 'page': page_num + 1}
                    ))
        return documents

    def _process_html_file(self, file_path: str, source_name: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._process_html_content(content, source_name)

    def _process_html_content(self, html_content: str, source_name: str) -> List[Document]:
        readable_doc = ReadabilityDocument(html_content)
        soup = BeautifulSoup(readable_doc.summary(), 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']) if p.get_text()]
        return [Document(page_content=clean_text(p), metadata={'source': source_name}) for p in paragraphs if len(p) > 50]

    def _process_markdown_file(self, file_path: str, source_name: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Simple split by paragraphs for markdown
        paragraphs = content.split('\n\n')
        return [Document(page_content=clean_text(p), metadata={'source': source_name}) for p in paragraphs if len(p) > 20]

    def _process_text_file(self, file_path: str, source_name: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        paragraphs = content.split('\n\n')
        return [Document(page_content=clean_text(p), metadata={'source': source_name}) for p in paragraphs if len(p) > 20]

