from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
import numpy as np
from typing import List, Tuple
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os

class RAGFusionSystem:
    def __init__(self):
        """Initialize the RAG Fusion system with required components"""
        self.llm = ChatOpenAI(temperature=0.1)
        self.tokenizer, self.model = self.load_model()
        
    @staticmethod
    def load_model():
        """Load the transformer model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()  # Set model to evaluation mode
        return tokenizer, model

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on transformer output
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask for the input
            
        Returns:
            torch.Tensor: Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for given texts
        
        Args:
            texts: List of input texts
            
        Returns:
            np.ndarray: Generated embeddings
        """
        # Ensure processing on CPU if no GPU available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(
                model_output,
                encoded_input['attention_mask']
            )
        
        return embeddings.cpu().numpy()

    def generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple search queries using RAG-Fusion approach
        
        Args:
            question: Original user question
            
        Returns:
            List[str]: List of generated queries
        """
        template = """You are an AI assistant that generates alternative search queries.
        Generate 3 different versions of the following question that capture the same meaning
        but use different wording or focus on different aspects:
        
        Question: {question}
        
        Output only the questions, one per line, without numbering or prefixes."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt 
            | self.llm 
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        
        # Get generated queries and clean them
        queries = chain.invoke({"question": question})
        queries = [q.strip() for q in queries if q.strip()]
        
        # Add original question and ensure uniqueness
        queries.append(question)
        return list(set(queries))  # Remove any duplicates

    def search_milvus(self, 
                     query_embedding: np.ndarray, 
                     collection: Collection, 
                     top_k: int = 3) -> List[dict]:
        """
        Search Milvus collection with query embedding
        
        Args:
            query_embedding: Query embedding vector
            collection: Milvus collection
            top_k: Number of results to return
            
        Returns:
            List[dict]: Search results
        """
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "id"]
        )
        
        return [{
            'text': hit.entity.get('text'),
            'score': float(hit.distance),  # Convert to float for JSON serialization
            'id': int(hit.entity.get('id'))
        } for hit in results[0]]

    def reciprocal_rank_fusion(self, 
                             results: List[List[dict]], 
                             k: int = 60) -> List[Tuple[dict, float]]:
        """
        Implement reciprocal rank fusion for multiple result lists
        
        Args:
            results: List of search results from different queries
            k: Constant for RRF calculation
            
        Returns:
            List[Tuple[dict, float]]: Reranked results with scores
        """
        fused_scores = {}
        
        for query_results in results:
            for rank, doc in enumerate(query_results):
                doc_id = str(doc['id'])  # Use ID as key instead of full document
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {'doc': doc, 'score': 0}
                fused_scores[doc_id]['score'] += 1 / (rank + k)
        
        # Sort by score and return documents
        return sorted(
            [(doc_info['doc'], doc_info['score']) 
             for doc_info in fused_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )

    def rag_fusion_search(self, 
                         question: str, 
                         collection: Collection, 
                         top_k: int = 3) -> Tuple[str, List[dict]]:
        """
        Perform RAG-Fusion search with multiple generated queries
        
        Args:
            question: User question
            collection: Milvus collection
            top_k: Number of results to return
            
        Returns:
            Tuple[str, List[dict]]: Combined context and detailed results
        """
        # Generate multiple queries
        queries = self.generate_queries(question)
        print(f"\nGenerated queries:\n" + "\n".join(queries))
        
        # Get results for each query
        all_query_results = []
        for query in queries:
            # Get embeddings for the query
            query_embedding = self.get_embeddings([query])[0]
            
            # Search Milvus
            query_results = self.search_milvus(query_embedding, collection, top_k)
            all_query_results.append(query_results)
        
        # Apply reciprocal rank fusion
        fused_results = self.reciprocal_rank_fusion(all_query_results)
        
        # Take top results
        final_results = [result[0] for result in fused_results[:top_k]]
        
        # Combine texts for context
        context = "\n\n---\n\n".join([r['text'] for r in final_results])
        
        return context, final_results

def setup_milvus(file_path: str = None) -> Collection:
    """
    Setup Milvus collection and optionally load data from file
    
    Args:
        file_path: Path to input text file
        
    Returns:
        Collection: Configured Milvus collection
    """
    db_file = "milvus_lite.db"
    collection_name = "text_chunks"
    
    # Connect to Milvus Lite
    connections.connect("default", uri=db_file)
    
    # Check if collection exists
    if os.path.exists(db_file) and utility.has_collection(collection_name):
        print("Loading existing collection...")
        collection = Collection(collection_name)
        collection.load()
        return collection
    
    if not file_path:
        raise ValueError("Need a file path to initialize new collection")
        
    # Initialize system and process document
    rag_system = RAGFusionSystem()
    
    # Read and process the document
    with open(file_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Split into chunks (implement your chunking logic here)
    chunk_size = 512
    overlap = 50
    words = document.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # Generate embeddings
    chunk_embeddings = rag_system.get_embeddings(chunks)
    
    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, "Text chunk collection")
    
    # Create and setup collection
    collection = Collection(collection_name, schema)
    
    # Create index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    
    # Insert data
    collection.insert([
        chunks,
        chunk_embeddings.tolist()
    ])
    
    collection.flush()
    collection.load()
    
    print(f"Initialized collection with {collection.num_entities} chunks")
    return collection

def main():
    parser = argparse.ArgumentParser(description="RAG Fusion Search System")
    parser.add_argument("--file", required=True, help="Path to the text file to initialize collection")
    args = parser.parse_args()
    
    # Initialize system
    rag_system = RAGFusionSystem()
    collection = setup_milvus(args.file)
    
    print("\nRAG Fusion Search System")
    print("Enter your questions or type 'quit' to exit.")
    
    try:
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            context, results = rag_system.rag_fusion_search(
                question=question,
                collection=collection,
                top_k=3
            )
            
            print("\nTop 3 most relevant chunks:")
            print("=" * 80)
            print(context)
            print("=" * 80)
            
            print("\nDetailed search results:")
            print(json.dumps(results, indent=2))
            
    finally:
        # Cleanup
        collection.release()
        utility.drop_collection(collection.name)
        connections.disconnect("default")

if __name__ == "__main__":
    main()