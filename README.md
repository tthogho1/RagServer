# ChatBot Server by Rag with Cloud
Embedding website data, registering it in Vector DB, and asking questions about it.

## config.ini
Create a config.ini file in advance. and store vectors to Qdrant Cloud

Create an account for openai and qdrant Prepare the account key in advance
If qdrant is not specified, use Chromadb　in memory and need to load data.

-- format --
[apikey]
openapikey = [apikey for openai]
qdrantkey = [apikey for qdrant]
qdranturl = [url for qdrant]

[vectordb]
name = Qdrant

## Environment construction using docker

1. git clone https://github.com/tthogho1/RagServer.git
2. cd RagServer
3. create config.ini file  
4. docker buld -t myimage .
5. docker run -it --name test -p 8000:8000 myimage

Access http://localhost8000 from your browser
