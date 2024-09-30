from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # BERT embedding size
    FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)
]
index_params = {
  "metric_type":"COSINE",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}

schema = CollectionSchema(fields=fields, description="Store BERT embeddings and their corresponding words")
collection_name = "word_embeddingsv3"

# Create the collection
collection = Collection(name=collection_name, schema=schema)
collection.create_index(
  field_name="embedding", 
  index_params=index_params
)

import torch
from transformers import AutoTokenizer, AutoModel

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get BERT embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Convert tensor to numpy

# Example words and their categories
categories = {
    "dress_type": ["T-shirt", "dress", "jacket", "shirt"],
    "material": ["cotton", "wool", "leather"],
    "fabric": ["stretch", "denim", "silk"],
    "gender": ["ladies", "mens"],
    "color": ["firered", "blue", "green", "black"],
    "size": ["M", "L", "S", "XL"]
}

category_embeddings = []

for category, words in categories.items():
    for word in words:
        category_embeddings.append(
            {"embedding":get_embedding(word),"word":word,"category":category})

# Insert the data into the collection
collection.insert(category_embeddings)

# # Load collection to enable searching
# collection.load()


# # Function to search the collection by embedding
# def search_embedding(query_text, top_k=1):
#     query_embedding = get_embedding(query_text).tolist()
    
#     search_params = {
#         "metric_type": "L2",  # L2 for Euclidean distance, can also use cosine similarity
#         "params": {"nprobe": 10}
#     }
    
#     results = collection.search(
#         data=[query_embedding],
#         anns_field="embedding",
#         param=search_params,
#         limit=top_k,
#         output_fields=["word", "category"]  # Retrieve both word and category
#     )
    
#     # Display search results
#     for result in results[0]:
#         print(f"Word: {result.entity.get('word')}, Category: {result.entity.get('category')}, Distance: {result.distance}")

#     result = results[0]
#     return  result.entity.get('word'), result.entity.get('category'),result.distance

# def classify_input_text(input_text):
#     words = input_text.split()
#     results = []
#     for word in words:
#         best_word, best_category,max_similarity = search_embedding(word)
#         results.append((word, best_word, best_category,max_similarity))
#     return results


# # Search for a word
# val = classify_input_text("T-shirt")
# print(val)
