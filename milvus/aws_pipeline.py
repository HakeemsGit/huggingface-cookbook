import io
import boto3
from PIL import Image
from langchain_community.document_loaders import S3DirectoryLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_openai import OpenAIEmbeddings
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema, connections, utility

# Initialize S3 client
s3_client = boto3.client('s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Initialize the text embedding model from OpenAI
text_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Initialize the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image_with_clip(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy()

def process_document(doc):
   source = doc.metadata['source']  # Extract document source (e.g., S3 URL)

   # Processing Text Files
   if source.endswith('.txt'):
       text = doc.page_content  # Extract the content from the text file
       print(f"Processing .txt file: {source}")
       return text, text_embedding_model.embed_documents([text])  # Convert to embeddings

   # Processing PDF Files
   elif source.endswith('.pdf'):
       content = doc.page_content  # Extract content from the PDF
       print(f"Processing .pdf file: {source}")
       return content, text_embedding_model.embed_documents([content])  # Convert to embeddings

   # Processing Image Files
   elif source.endswith('.png'):
       print(f"Processing .png file: {source}")
       bucket_name, object_key = parse_s3_url(source)  # Parse the S3 URL
       response = s3_client.get_object(Bucket=bucket_name, Key=object_key)  # Fetch image from S3
       img_bytes = response['Body'].read()

       # Load the image and convert to embeddings
       img = Image.open(io.BytesIO(img_bytes))
       return source, embed_image_with_clip(img)  # Convert to image embeddings

def parse_s3_url(s3_url):
    parts = s3_url.replace("s3://", "").split("/", 1)
    bucket_name = parts[0]
    object_key = parts[1]
    return bucket_name, object_key


def create_collection(collection_name):
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # Drop collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="review", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="product_image", dtype=DataType.VARCHAR, max_length=65535)
    ]
    
    # Create collection schema
    schema = CollectionSchema(fields=fields, description="Product data with text and image embeddings")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for text embeddings
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="text_embedding", index_params=index_params)
    collection.create_index(field_name="image_embedding", index_params=index_params)
    
    return collection

create_collection("products-data")

def ingest_data(collection, points):
    """Insert data into Milvus collection"""
    ids = [p["id"] for p in points]
    text_embeddings = [p["vector"]["text_embedding"] for p in points]
    image_embeddings = [p["vector"]["image_embedding"] for p in points]
    reviews = [p["payload"]["review"] for p in points]
    product_images = [p["payload"]["product_image"] for p in points]
    
    data = [
        ids,
        text_embeddings,
        image_embeddings,
        reviews,
        product_images
    ]
    
    collection.insert(data)
    collection.flush()
    return f"Inserted {len(ids)} records successfully"

if __name__ == "__main__":
    collection_name = "products-data"
    collection = create_collection(collection_name)
    
    points = []
    for i in range(1,6): # Five documents
        folder = f"p_{i}"
        loader = S3DirectoryLoader(
            "product-dataset",
            folder,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        docs = loader.load()
        text_embedding, image_embedding, text_review, product_image = None, None, "", ""
        
        for idx, doc in enumerate(docs):
            source = doc.metadata['source']
            if source.endswith(".txt"):
                text_review, text_embedding = process_document(doc)
            elif source.endswith(".png"):
                product_image, image_embedding = process_document(doc)
                
        if text_review and text_embedding is not None and image_embedding is not None:
            point = {
                "id": i,  # Using document number as ID
                "vector": {
                    "text_embedding": text_embedding[0],
                    "image_embedding": image_embedding[0].tolist()
                },
                "payload": {
                    "review": text_review,
                    "product_image": product_image
                }
            }
            points.append(point)
    
    operation_info = ingest_data(collection, points)
    print(operation_info)
    
    # Clean up
    connections.disconnect("default")
