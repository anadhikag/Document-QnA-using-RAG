# Create document ingestion module

document_processor_content = '''
import os
import re
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

# Document parsing imports
import pymupdf  # PyMuPDF for PDFs
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

from ..utils.helpers import clean_text, extract_page_number

@dataclass
class Document:
    """Represents a document with content and metadata"""
    page_content: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Clean the content
        self.page_content = clean_text(self.page_content)

class DocumentProcessor:
    """Handles ingestion of various document formats"""
    
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def ingest_file(self, file_path: str) -> List[Document]:
        """Process a file and return list of documents"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size_bytes:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_mb}MB")
        
        # Determine file type and process accordingly
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path)
        elif file_ext in ['.html', '.htm']:
            return self._process_html_file(file_path)
        elif file_ext in ['.md', '.markdown']:
            return self._process_markdown_file(file_path)
        elif file_ext in ['.txt']:
            return self._process_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def ingest_url(self, url: str) -> List[Document]:
        """Process content from a URL"""
        
        try:
            # Fetch the content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check content size
            if len(response.content) > self.max_file_size_bytes:
                raise ValueError(f"Content too large: {len(response.content) / 1024 / 1024:.1f}MB > {self.max_file_size_mb}MB")
            
            # Process HTML content
            return self._process_html_content(response.text, url)
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL {url}: {str(e)}")
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file using PyMuPDF"""
        
        documents = []
        
        try:
            with pymupdf.open(file_path) as pdf_doc:
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    
                    if text.strip():  # Only process non-empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': os.path.basename(file_path),
                                'page': page_num + 1,
                                'type': 'pdf',
                                'file_path': file_path
                            }
                        )
                        documents.append(doc)
        
        except Exception as e:
            raise ValueError(f"Failed to process PDF {file_path}: {str(e)}")
        
        return documents
    
    def _process_html_file(self, file_path: str) -> List[Document]:
        """Process HTML file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self._process_html_content(content, file_path)
    
    def _process_html_content(self, html_content: str, source: str) -> List[Document]:
        """Process HTML content using readability and BeautifulSoup"""
        
        documents = []
        
        try:
            # Use readability to extract main content
            doc = ReadabilityDocument(html_content)
            main_content = doc.content()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(main_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text and split by paragraphs
            text = soup.get_text()
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
            
            # Create documents for each substantial paragraph
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Only substantial paragraphs
                    doc = Document(
                        page_content=paragraph,
                        metadata={
                            'source': os.path.basename(source) if os.path.isfile(source) else source,
                            'paragraph': i + 1,
                            'type': 'html',
                            'url': source if source.startswith('http') else None
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            raise ValueError(f"Failed to process HTML content: {str(e)}")
        
        return documents
    
    def _process_markdown_file(self, file_path: str) -> List[Document]:
        """Process Markdown file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by headers and paragraphs
        sections = self._split_markdown_content(content)
        
        documents = []
        for i, section in enumerate(sections):
            if section.strip():
                doc = Document(
                    page_content=section,
                    metadata={
                        'source': os.path.basename(file_path),
                        'section': i + 1,
                        'type': 'markdown',
                        'file_path': file_path
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _process_text_file(self, file_path: str) -> List[Document]:
        """Process plain text file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
        
        documents = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph and len(paragraph) > 20:  # Only substantial paragraphs
                doc = Document(
                    page_content=paragraph,
                    metadata={
                        'source': os.path.basename(file_path),
                        'paragraph': i + 1,
                        'type': 'text',
                        'file_path': file_path
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _split_markdown_content(self, content: str) -> List[str]:
        """Split markdown content by headers"""
        
        # Split by headers
        sections = re.split(r'\\n(?=#+\\s)', content)
        
        # Further split long sections by paragraphs
        final_sections = []
        for section in sections:
            if len(section) > 2000:  # Split long sections
                paragraphs = section.split('\\n\\n')
                final_sections.extend(paragraphs)
            else:
                final_sections.append(section)
        
        return [s.strip() for s in final_sections if s.strip()]
'''

# Write document processor
with open(os.path.join("document_qna", "src", "ingestion", "document_processor.py"), "w") as f:
    f.write(document_processor_content.strip())

print("✅ Created ingestion/document_processor.py")