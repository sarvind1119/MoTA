from langchain.embeddings.openai import OpenAIEmbeddings
import os
from pinecone import ServerlessSpec
#from main import *
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

index_name = 'askmydoc2'

# get openai api key from platform.openai.com
OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# vector_store.similarity_search(
#     query,  # our search query
#     k=3  # return 3 most relevant docs
# )

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
query="Give the key points of TwelfthFiveYearPlan2012-17"
#query = "who is Arvind Kejriwal?"

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=True)

    answer = chain.invoke(q)
    answer = ask_and_get_answer(vectorstore_from_docs, q)
    print(answer['result'])
    print('Reference:\n')
    for x in range(len(answer["source_documents"][0].metadata)):
        raw_dict = answer["source_documents"][x].metadata
        print("Page number:", raw_dict['page'], "Filename:", raw_dict['source'])
    return answer
# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
query="Give the key points of TwelfthFiveYearPlan2012-17"
#qa_with_sources(query)

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on MoTA Twelfth Five Year Plan(2012–2017) Social Sectors...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to annual reports of MEITY from year 2017 to 2023



    [Documents Repository](https://drive.google.com/file/d/1FKRqgTYPW5BdjTGiLbSyvFbNNshMXiB3/view?usp=drive_link)
 
    ''')
    #add_vertical_space(5)
    st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')

def main():
    #st.title("Question and Answering App powered by LLM and Pinecone")

    text_input = st.text_input("Ask your query...") 
    if st.button("Ask Query"):
        if len(text_input)>0:
            #st.info("Your Query: " + text_input)
            #answer = qa_with_sources(text_input)
            #st.success(answer)
            answer = ask_and_get_answer(vectorstore,text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
