from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

def getResponse(query):
    
    llm = ChatOpenAI(model="gpt-4o")

    DATA_DIR = './data'
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt")

    docs = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # text_splitter1 = CharacterTextSplitter(chunk_size=386,chunk_overlap=0)
    # text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text_splitter3 = SemanticChunker(embeddings, breakpoint_threshold_type="percentile"
)
    split_doc = text_splitter3.split_documents(docs)
    
    db = FAISS.from_documents(split_doc, embeddings)
    retriever = db.as_retriever()

    setup = RunnableParallel(
        {
            "context": retriever, "user_question": RunnablePassthrough()
        }
    )

    template =  """
    Answer the question based on the context below. Keep the answer short and concise.
    Briefly outline the logic behind your answer.
    Context: {context}
    Question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain =  setup | prompt | llm | StrOutputParser()
    return chain.invoke(query)


st.set_page_config()
st.title("🧑‍⚖️HSP ChatBot")

prompt = st.chat_input("What would you like to know?")


if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    with st.chat_message("AI"):
        response = getResponse(prompt)
        st.write(response)