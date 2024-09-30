from pymilvus import connections, Collection

import torch
from transformers import AutoTokenizer, AutoModel
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection_name = "word_embeddingsv3"

# Create the collection
collection = Collection(name=collection_name)
# Load collection to enable searching
collection.load()

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


# Function to search the collection by embedding
def search_embedding(query_text, top_k=1):
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

def classify_input_text(input_text):
    words = input_text.split()
    results = []
    for word in words:
        best_word, best_category,max_similarity = search_embedding(word)
        results.append((word, best_word, best_category,max_similarity))
    return results


# Search for a word
val = classify_input_text("OX Pro Crew Neck Sweatshirt Small Logo - Grey Marl - Small")
for word, best_word, best_category,max_similarity in val:
    print(f"Original Word - {word}, Best Word: {best_word}, Category: {best_category}, Distance: {max_similarity}")
