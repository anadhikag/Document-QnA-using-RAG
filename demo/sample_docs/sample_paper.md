# Sample Research Paper

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

from document_qna import DocumentProcessor
processor = DocumentProcessor()
documents = processor.ingest_file("paper.pdf")

## Results

The system achieves good performance on question answering tasks while maintaining low memory usage. Example questions include:
- "What is the main contribution?"
- "How does the chunking work?"
- "What embedding model is used?"

## Conclusion

This demonstrates a practical approach to building local RAG systems that can run on consumer hardware.
