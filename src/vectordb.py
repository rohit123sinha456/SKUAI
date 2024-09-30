from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os
load_dotenv()
# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_collection():
    # Connect to Milvus
    connections.connect("default", host=os.getenv('MILVUSURL'), port="19530")

    # Define the schema for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # BERT embedding size
        FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)
    ]

    schema = CollectionSchema(fields=fields, description="Store BERT embeddings and their corresponding words and categories")
    collection_name = os.getenv('MILVUSCOLLECTION')

    # Check if collection exists
    if utility.has_collection(collection_name):
        print(f"Loading existing collection: {collection_name}")
        collection = Collection(name=collection_name)  # Load the collection
    else:
        print(f"Creating new collection: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)  # Create the collection

        # Create the index on the embedding field
        index_params = {
        "metric_type":"COSINE",
        "index_type":"IVF_FLAT",
        "params":{"nlist":1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    # Load the collection into memory
    collection.load()

    return collection

# Function to get BERT embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Convert tensor to numpy

def insert_categories(collection,categories):
    category_embeddings = []
    for category, words in categories.items():
        for word in words:
            category_embeddings.append(
                {"embedding":get_embedding(word),"word":word,"category":category})
    try:
        collection.insert(category_embeddings)
    except Exception as e:
        return str(e)
    
    return None


# Function to search the collection by embedding
def search_embedding(collection,query_text, top_k=1):
    query_embedding = get_embedding(query_text).tolist()
    
    search_params = {
        "metric_type": "COSINE",  # L2 for Euclidean distance, can also use cosine similarity
        "params": {"nprobe": 10}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["word", "category"]  # Retrieve both word and category
    )
    
    # Display search results
    # for result in results[0]:
    #     print(f"Word: {result.entity.get('word')}, Category: {result.entity.get('category')}, Distance: {result.distance}")

    result = results[0][0]
    return  result.entity.get('word'), result.entity.get('category'),result.distance

def classify_input_text(collection,input_text):
    words = input_text.split()
    results = []
    for word in words:
        best_word, best_category,max_similarity = search_embedding(collection,word)
        results.append((word, best_word, best_category,max_similarity))
    return results

