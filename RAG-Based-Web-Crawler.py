 import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging as hf_logging
import re
import collections

hf_logging.set_verbosity_error()

class RAGSystem:

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = pipeline('question-answering', model='deepset/roberta-base-squad2')
        self.documents = []
        self.vector_store = None
        self.doc_source_map = {}

    def crawl_site(self, start_url, max_pages=30):
        domain = urlparse(start_url).netloc
        queue = collections.deque([start_url])
        visited_urls = {start_url}
        crawled_pages = []
        crawl_delay = 1.0

        print(f"Starting crawl on domain: {domain}. Will fetch up to {max_pages} pages.")

        while queue and len(crawled_pages) < max_pages:
            current_url = queue.popleft()
            print(f"Crawling [{len(crawled_pages) + 1}/{max_pages}]: {current_url}")

            try:
                time.sleep(crawl_delay)
                response = requests.get(current_url, timeout=15, headers={'User-Agent': 'RAGSystemBot/1.0'})
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                raw_text = ' '.join(soup.stripped_strings)
                clean_text = re.sub(r'\s+', ' ', raw_text).strip()

                if clean_text:
                    crawled_pages.append({'url': current_url, 'text': clean_text})

                for link_tag in soup.find_all('a', href=True):
                    absolute_link = urljoin(current_url, link_tag['href'])
                    parsed_link = urlparse(absolute_link)

                    link_url = parsed_link._replace(fragment="", query="").geturl()

                    if urlparse(link_url).netloc == domain and link_url not in visited_urls:
                        visited_urls.add(link_url)
                        queue.append(link_url)

            except requests.RequestException as e:
                print(f"Error crawling {current_url}: {e}")

        return crawled_pages

    def chunk_text(self, text, chunk_size=500, overlap=100):
        words = text.split()
        if not words:
            return []
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def build_index(self, crawled_content):
        text_chunks = []
        for doc in crawled_content:
            chunks = self.chunk_text(doc['text'])
            for chunk in chunks:
                text_chunks.append(chunk)
                self.doc_source_map[len(text_chunks)-1] = doc['url']

        self.documents = text_chunks
        if not self.documents:
            print("No text chunks were generated. Index cannot be built.")
            return

        print(f"Embedding {len(self.documents)} text chunks...")
        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(np.array(embeddings, dtype=np.float32))
        print("Vector index built successfully.")

    def retrieve_context(self, query, top_k=5):
        if not self.vector_store:
            return [], []

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), top_k)

        # Filter out -1 indices which indicate no result
        valid_indices = [i for i in indices[0] if i != -1]

        retrieved_chunks = [self.documents[i] for i in valid_indices]
        retrieved_sources = [self.doc_source_map[i] for i in valid_indices]

        return retrieved_chunks, list(set(retrieved_sources))

    def generate_answer(self, query):
        if self.vector_store is None:
            return "The system is not ready. Please crawl a site first.", []

        retrieved_chunks, sources = self.retrieve_context(query)

        if not retrieved_chunks:
            return "I could not find any relevant information on the crawled site.", []

        context = " ".join(retrieved_chunks)

        result = self.generator(question=query, context=context)

        if result['score'] < 0.05:
            return "I could not find a confident answer in the indexed content.", sources
        else:
            return result['answer'], sources

    def start_system(self):
        print("--- Retrieval-Augmented Generation System ---")

        while True:
            base_url = input("Enter the full URL of the site to crawl (e.g., https://web.mit.edu/): ")
            if base_url.lower() == 'quit': return
            if urlparse(base_url).scheme and urlparse(base_url).netloc:
                break
            else:
                print("Invalid URL format. Please provide a full URL like 'https://example.com'.")

        crawled_data = self.crawl_site(base_url)
        if not crawled_data:
            print("Crawling did not yield any content. The system cannot proceed.")
            return

        self.build_index(crawled_data)
        print("\nIndexing complete. You may now ask questions based on the crawled content.")
        print("Type 'quit' to exit.")

        while True:
            user_query = input("\nYour Question: ")
            if user_query.lower() == 'quit':
                break

            answer, sources = self.generate_answer(user_query)
            print(f"\nAnswer: {answer}")
            if sources:
                print("\nSources:")
                for url in sources:
                    print(f"- {url}")

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.start_system()