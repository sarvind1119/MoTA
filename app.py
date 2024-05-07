from langchain.embeddings.openai import OpenAIEmbeddings
import os
from pinecone import ServerlessSpec
import os
from pinecone import Pinecone
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'mota1'

# get openai api key from platform.openai.com
OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-3-small '

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

#query = "who is Arvind Kejriwal?"

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
# )
llm = OpenAI()
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )

# qa.run(query)

from langchain.chains import RetrievalQAWithSourcesChain
query = "Summarise yourself"
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
qa_with_sources(query)

# from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
# llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# chain = load_qa_chain(llm, chain_type="stuff")
# query = "Explain in detail Post Matric Scholarship"
# docs = vectorstore.similarity_search(query)
# chain.run(input_documents=docs, question=query)

# def ask_and_get_answer(vector_store, q, k=3):
#     from langchain.chains import RetrievalQA
#     from langchain_openai import ChatOpenAI

#     llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

#     retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

#     chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     answer = chain.invoke(q)
#     return answer

# q = "Summarise yourself"
# answer = ask_and_get_answer(vectorstore, q)
# print(answer)