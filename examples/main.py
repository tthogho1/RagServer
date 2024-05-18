#import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings

from openai import OpenAI

# 1. Load the data
#loader = WebBaseLoader(
#    web_paths=("https://docs.smith.langchain.com",),
#    bs_kwargs=dict(
#        parse_only=bs4.SoupStrainer(
#            class_=("post-content", "post-title", "post-header")
#        )
#    ),
#)
loader = WebBaseLoader("https://docs.smith.langchain.com")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Create Ollama embeddings and vector store
embeddings = OpenAIEmbeddings(
    openai_api_key=""
)

#vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
vectorstore = Chroma(embedding_function=embeddings)
vectorstore.add_documents(documents=splits)

client = OpenAI(
    api_key=""
)
def openai_llm(question, context):
    messages = [
        {"role": "system", "content": question},
        {"role": "user", "content": context},
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

# 4. RAG Setup
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return openai_llm(question, formatted_context)

# 5. Use the RAG App
result = rag_chain("how to use langchain?")
print(result)