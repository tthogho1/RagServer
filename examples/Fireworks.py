from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.llms.fireworks import Fireworks
from IPython.display import Markdown, display
import chromadb

from llama_index.llms.fireworks import Fireworks
from llama_index.embeddings.fireworks import FireworksEmbedding

fw_api_key = ""

llm = Fireworks(
    api_key=fw_api_key,
    temperature=0, model="accounts/fireworks/models/mixtral-8x7b-instruct"
)
# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = FireworksEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

# load documents
documents = SimpleDirectoryReader("./data/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query Data
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"{response}"))