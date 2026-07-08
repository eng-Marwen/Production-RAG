import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import chromadb
from langchain_chroma import Chroma

def chunksToEmbeddings(chunks):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )
    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )   
    import chromadb

    client = chromadb.CloudClient(
        api_key='ck-Fx3PHuVGmyASkJ7cwZBwTcswwxnpvH6TPWLTa1d52UAE',
        tenant='0a9dec27-3647-4676-8611-8de5eb3a8c79',
        database='prototype'
    )


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="database_rag"
    )
    return embeddings