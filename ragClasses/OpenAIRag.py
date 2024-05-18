import bs4
import configparser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings

from openai import OpenAI

class OpenAIRag:
    config = configparser.ConfigParser()
    config.read('config.ini')
    apikey = config['apikey']['openapikey']
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=apikey
    )
    vectorstore = Chroma(embedding_function=embeddings)
    client = OpenAI(
        api_key=apikey
    )

    def __init__(self):
        print("create intstance openai")
        
    # インスタンスメソッド
    def setUrl (self,urlString):
        self.loader = WebBaseLoader(urlString)
    
    def loadUrl(self):        
        self.docs = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = text_splitter.split_documents(self.docs)
        self.vectorstore.add_documents(documents=self.splits)
        self.retriever = self.vectorstore.as_retriever()
            
    def openai_llm(self,question, context):
        messages = [
            {"role": "system", "content": question},
            {"role": "user", "content": context},
        ]
    
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        response_message = response.choices[0].message.content
        
        print(response_message)
        return response_message

    def combine_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def rag_chain(self,question):
        retrieved_docs = self.retriever.invoke(question)
        formatted_context = self.combine_docs(retrieved_docs)
        
        return self.openai_llm(question, formatted_context)