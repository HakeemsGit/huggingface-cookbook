import time
from sentence_transformers import SentenceTransformer
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pathlib import Path
from tqdm import tqdm
from uuid import uuid4

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """Embed a list of documents using the model."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Embed a single query using the model."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings()

def create_milvus_store(collection_name="speed_test"):
    """Initialize a fresh Milvus store using LangChain."""
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": "milvus_lite.db"},
        drop_old=True  # This ensures we start fresh
    )

def load_documents(books_dir):
    """Load all books as LangChain Documents."""
    documents = []
    books_path = Path(books_dir)
    
    print("Loading books...")
    for book_file in books_path.glob("*.txt"):
        print(f"Reading {book_file.name}")
        with open(book_file, 'r', encoding='utf-8') as file:
            content = file.read()
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            # Create Document objects with metadata
            docs = [
                Document(
                    page_content=p,
                    metadata={"source": book_file.name}
                ) for p in paragraphs
            ]
            documents.extend(docs)
    
    return documents

def test_single_insertions(documents):
    """Test inserting documents one at a time."""
    vector_store = create_milvus_store("single_test")
    start_time = time.time()
    
    for doc in tqdm(documents, desc="Single insertions"):
        vector_store.add_documents([doc], ids=[str(uuid4())])
    
    return time.time() - start_time

def test_bulk_insertion(documents):
    """Test inserting all documents at once."""
    vector_store = create_milvus_store("bulk_test")
    start_time = time.time()
    
    # Generate UUIDs for all documents
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents, ids=uuids)
    
    return time.time() - start_time

def test_batched_insertion(documents, batch_size=100):
    """Test inserting documents in batches."""
    vector_store = create_milvus_store(f"batch_test_{batch_size}")
    start_time = time.time()
    
    for i in tqdm(range(0, len(documents), batch_size), desc=f"Batch insertions (size={batch_size})"):
        batch = documents[i:i + batch_size]
        uuids = [str(uuid4()) for _ in range(len(batch))]
        vector_store.add_documents(batch, ids=uuids)
    
    return time.time() - start_time

def main():
    print("LangChain Milvus Insertion Speed Test")
    print("====================================")
    
    # Load documents
    documents = load_documents('books')
    num_documents = len(documents)
    print(f"\nTotal documents to process: {num_documents}")
    
    # Test single insertions
    print("\nTesting single insertions...")
    single_time = test_single_insertions(documents)
    
    # Test bulk insertion
    print("\nTesting bulk insertion...")
    bulk_time = test_bulk_insertion(documents)
    
    # Test different batch sizes
    batch_sizes = [100, 500, 1000]
    batch_times = {}
    for batch_size in batch_sizes:
        print(f"\nTesting batch insertion (size={batch_size})...")
        batch_times[batch_size] = test_batched_insertion(documents, batch_size)
    
    # Print results
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"\nSingle insertions:")
    print(f"- Total time: {single_time:.2f} seconds")
    print(f"- Speed: {num_documents/single_time:.2f} documents/second")
    
    print(f"\nBulk insertion:")
    print(f"- Total time: {bulk_time:.2f} seconds")
    print(f"- Speed: {num_documents/bulk_time:.2f} documents/second")
    print(f"- Speedup vs single: {single_time/bulk_time:.2f}x")
    
    print("\nBatch insertions:")
    for batch_size, batch_time in batch_times.items():
        print(f"\nBatch size {batch_size}:")
        print(f"- Total time: {batch_time:.2f} seconds")
        print(f"- Speed: {num_documents/batch_time:.2f} documents/second")
        print(f"- Speedup vs single: {single_time/batch_time:.2f}x")
        print(f"- Speedup vs bulk: {bulk_time/batch_time:.2f}x")

if __name__ == "__main__":
    main()