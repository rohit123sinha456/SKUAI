from pymilvus import connections, Collection

import torch
from transformers import AutoTokenizer, AutoModel
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection_name = "word_embeddingsv3"

# Create the collection
collection = Collection(name=collection_name)
print(collection.name)
# Load collection to enable searching