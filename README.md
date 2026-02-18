RAG-Based Web Crawler & QA System

A lightweight Retrieval-Augmented Generation (RAG) system that:

-   Crawls a website
-   Extracts and cleans text content
-   Splits content into semantic chunks
-   Embeds using Sentence Transformers
-   Indexes with FAISS
-   Answers questions using a QA transformer model

Built entirely in Python — simple, local, and extensible.

Overview

Website → Crawl → Clean → Chunk → Embed → FAISS Index → Retrieve →
Answer

Unlike LLM-only systems, this approach: - Grounds answers in crawled
content - Reduces hallucination risk - Returns sources for
transparency - Works fully locally (no external APIs required)

Architecture

1)  Crawling

-   Breadth-first crawl within the same domain
-   Removes script, style, nav, header, footer tags
-   Cleans whitespace and normalizes text
-   Default: max 30 pages

2)  Chunking Strategy

-   Default chunk size: 500 words
-   Overlap: 100 words
-   Prevents context fragmentation
-   Improves retrieval continuity

3)  Embeddings Model used:

-   all-MiniLM-L6-v2 (SentenceTransformers)

Why? - Fast - Lightweight (~80MB) - Strong semantic similarity
performance

4)  Vector Store Library:

-   FAISS (IndexFlatL2)

Why? - Extremely fast similarity search - Memory-efficient - Ideal for
local RAG systems

5)  Answer Generation QA Model:

-   deepset/roberta-base-squad2

Pipeline: - Retrieve top-k chunks - Concatenate into context - Extract
answer span - Apply confidence threshold (score < 0.05 filtered)

Installation

pip install requests beautifulsoup4 numpy faiss-cpu
sentence-transformers transformers

Usage

python RAG-Based-Web-Crawler.py

Steps: 1. Enter full website URL 2. Wait for crawling and indexing 3.
Start asking questions 4. Type ‘quit’ to exit

Class Structure

RAGSystem

-   crawl_site(): Domain-restricted BFS crawler
-   chunk_text(): Sliding-window chunking
-   build_index(): Embedding + FAISS indexing
-   retrieve_context(): Semantic similarity search
-   generate_answer(): QA-based answer extraction
-   start_system(): Interactive CLI interface

Configuration Points

Crawl depth: crawl_site(start_url, max_pages=30)

Chunking: chunk_text(chunk_size=500, overlap=100)

Retrieval depth: retrieve_context(top_k=5)

Confidence threshold: if result[‘score’] < 0.05

Strengths

-   Fully local
-   No paid APIs required
-   Fast embedding
-   Source attribution
-   Easy to extend

Limitations

-   Extractive QA model (not generative)
-   No re-ranking
-   No persistent index saving
-   No multi-turn memory
-   Basic HTML cleaning

Potential Improvements

Performance: - Batch embeddings - Async crawling - Persistent FAISS
index - Parallel requests

Retrieval: - Add cross-encoder re-ranking - Hybrid search (BM25 +
dense) - Metadata filtering

Chunking: - Token-based chunking - Recursive splitting - Semantic
chunking

Generation: - Replace QA model with Llama, Mistral, or GPT-style
generative model

UX: - Streamlit frontend - FastAPI backend - Docker containerization

Project Structure

. ├── RAG-Based-Web-Crawler.py └── README.txt

Final Thoughts

This project demonstrates how a complete RAG system can be built using:

-   Traditional crawling
-   Modern embeddings
-   Vector similarity search
-   Transformer-based QA

All within a single Python script.
