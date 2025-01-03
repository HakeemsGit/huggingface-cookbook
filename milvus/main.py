import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import textwrap
import os

# Load model
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Get embeddings
def get_embeddings(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).numpy()

# Chunk text
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Text Similarity Checker using Milvus Lite")
    parser.add_argument("file", help="Path to the text file")
    args = parser.parse_args()

    # Load the model
    tokenizer, model = load_model()

    # Connect to Milvus Lite
    db_file = "milvus_lite.db"
    connections.connect("default", uri=db_file)

    collection_name = "text_chunks"

    # Check if the database file and collection exist
    if os.path.exists(db_file) and utility.has_collection(collection_name):
        print("Database and collection already exist. Loading existing collection.")
        collection = Collection(collection_name)
        collection.load()
    else:
        # Read the text file
        with open(args.file, 'r') as f:
            document = f.read()

        # Chunk the document
        chunks = chunk_text(document)

        # Get embeddings for chunks
        chunk_embeddings = get_embeddings(chunks, tokenizer, model)

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, "Text chunk collection")

        # Create collection
        collection = Collection(collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)

        # Insert data
        collection.insert([chunks, chunk_embeddings.tolist()])
        collection.flush()

        # Load collection
        collection.load()

        print(f"Inserted {collection.num_entities} chunks into Milvus Lite")

    print("\nText Similarity Checker")
    print("Enter your queries or type 'quit' to exit.")

    while True:
        # Get user query
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        # Get query embedding
        query_embedding = get_embeddings([query], tokenizer, model)

        # Search in Milvus
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding[0].tolist()],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text"]
        )

        print(f"\nQuery: {query}")
        print("\nTop 3 similar chunks:")
        for i, hit in enumerate(results[0]):
            print(f"\nChunk {i+1}:")
            print("-" * 100)
            print(textwrap.fill(hit.entity.get('text'), width=100))
            print("-" * 100)

        print("\n" + "="*100 + "\n")

    # Disconnect from Milvus
    utility.drop_collection(collection_name)
    connections.disconnect("default")

if __name__ == "__main__":
    main()