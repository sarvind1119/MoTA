# Import necessary libraries
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import streamlit as st
import openai
import os
from pinecone import Pinecone

# Set directory path
directory = 'Docs/'

# Function to read documents from a directory
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Read documents from the specified directory
doc = read_doc(directory)

# Print the number of documents loaded
print(len(doc))

# Function to split data into chunks
def chunk_data(data, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    chunks = text_splitter.split_documents(data)
    return chunks

# Split documents into smaller chunks
docs = chunk_data(doc)

# Print the number of chunks created
print(len(docs))

# Print content of a specific chunk (e.g., chunk 9)
print(docs[8].page_content)

# Initialize OpenAI embeddings model
embeddings = OpenAIEmbeddings(model_name="ada")

# Initialize Pinecone client
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

# Function to insert or fetch embeddings from Pinecone index
def insert_or_fetch_embeddings(index_name, chunks):
    # Initialize Pinecone client and OpenAI embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Check if index exists
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        # Load embeddings from existing index
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # Create new index if it doesn't exist
        print(f'Creating index {index_name} and embeddings ...', end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Insert embeddings into the new index
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store

# Insert or fetch embeddings for the specified index name and chunks
vector_store = insert_or_fetch_embeddings(index_name='mota1', chunks=docs)

#-----SAME code as above without comments------
# from langchain.document_loaders import DirectoryLoader
# import pinecone
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI
# import streamlit as st

# directory = 'Docs/'

# def read_doc(directory):
#     file_loader=PyPDFDirectoryLoader(directory)
#     documents=file_loader.load()
#     return documents

# doc=read_doc(directory)
# len(doc)
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def chunk_data(data, chunk_size=500):
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
#     chunks = text_splitter.split_documents(data)
#     return chunks

# docs = chunk_data(doc)
# print(len(docs))


# print(docs[8].page_content)


# import openai
# from langchain.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model_name="ada")

# # query_result = embeddings.embed_query("Hello world")
# # len(query_result)

# import os
# from pinecone import Pinecone

# # initialize connection to pinecone (get API key at app.pinecone.io)
# api_key = os.environ.get('PINECONE_API_KEY')

# # configure client
# pc = Pinecone(api_key=api_key)


# def insert_or_fetch_embeddings(index_name, chunks):
#     # importing the necessary libraries and initializing the Pinecone client
#     import pinecone
#     from langchain_community.vectorstores import Pinecone
#     from langchain_openai import OpenAIEmbeddings
#     from pinecone import PodSpec

#     from pinecone import ServerlessSpec

      
#     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

#     # loading from existing index
#     if index_name in pc.list_indexes().names():
#         print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
#         vector_store = Pinecone.from_existing_index(index_name, embeddings)
#         print('Ok')
#     else:
#         # creating the index and embedding the chunks into the index 
#         print(f'Creating index {index_name} and embeddings ...', end='')

#         # creating a new index
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric='cosine',
#             spec=ServerlessSpec(cloud='aws', region='us-east-1')
#         )

#         # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
#         # inserting the embeddings into the index and returning a new Pinecone vector store object. 
#         vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
#         print('Ok')
        
#     return vector_store

# vector_store = insert_or_fetch_embeddings(index_name='mota1', chunks=docs)
