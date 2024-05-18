from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import openai
from typing import List
from tqdm import tqdm
from datetime import datetime

copy uri from MongoDB Atlas Cloud Console https://cloud.mongodb.com
uri ="mongodb+srv:/[user]:[password]@serverlessinstance0.yttctlp.mongodb.net/?retryWrites=true&w=majority&appName=ServerlessInstance0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

fw_client = openai.OpenAI(
    api_key="", # you can find Fireworks API key under accounts -> API keys
    base_url="https://api.fireworks.ai/inference/v1"
)

def generate_embeddings(input_texts: str, model_api_string: str, prefix="") -> List[float]:
    if prefix:
        input_texts = [prefix + text for text in input_texts]
    return fw_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    ).data[0].embedding
    
embedding_model_string = 'nomic-ai/nomic-embed-text-v1.5'
vector_database_field_name = 'embed' # define your embedding field name.
NUM_DOC_LIMIT = 2000 # the number of documents you will process and generate embeddings.

#sample_output = generate_embeddings(["This is a test."], embedding_model_string)
#print(f"Embedding size is: {str(len(sample_output))}")

db = client.sample_mflix # loading sample dataset from MongoDB Atlas
collection = db.movies
""""
keys_to_extract = ["plot", "genre", "cast", "title", "fullplot", "countries", "directors"]
for doc in tqdm(collection.find(
        {
            "fullplot":{"$exists": True},
            "released": { "$gt": datetime(2000, 1, 1, 0, 0, 0)},
        }
    ).limit(NUM_DOC_LIMIT), desc="Document Processing "):
    extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
    if vector_database_field_name not in doc:
        doc[vector_database_field_name] = generate_embeddings([extracted_str], embedding_model_string, "search_document: ")
    collection.replace_one({'_id': doc['_id']}, doc)
"""
"""
{
    "fields": [
        {
        "type": "vector",
        "path": "embed",
        "numDimensions": 768,
        "similarity": "dotProduct"
        }
    ]
}

"""

# Example query.
query = "I like Christmas movies, any recommendations?"
prefix="search_query: "
query_emb = generate_embeddings([query], embedding_model_string, prefix=prefix)

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_emb,
            "path": vector_database_field_name,
            "numCandidates": 100, # this should be 10-20x the limit
            "limit": 10, # the number of documents to return in the results
            "index": 'movie', # the index name you used in the earlier step
        }
    }
])
results_as_dict = {doc['title']: doc for doc in results}

print(f"From your query \"{query}\", the following movie listings were found:\n")
print("\n".join([str(i+1) + ". " + name for (i, name) in enumerate(results_as_dict.keys())]))


your_task_prompt = (
    "From the given movie listing data, choose a few great movie recommendation given the user query. "
    f"User query: {query}"
)

listing_data = ""
for doc in results_as_dict.values():
    listing_data += f"Movie title: {doc['title']}\n"
    for (k, v) in doc.items():
        if not(k in keys_to_extract) or ("embedding" in k): continue
        if k == "name": continue
        listing_data += k + ": " + str(v) + "\n"
    listing_data += "\n"

augmented_prompt = (augmented_prompt
    "movie listing data:\n"
    f"{listing_data}\n\n"
    f"{your_task_prompt}"
)

response = fw_client.chat.completions.create(
    messages=[{"role": "user", "content": augmented_prompt}],
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
)

print(response.choices[0].message.content)
