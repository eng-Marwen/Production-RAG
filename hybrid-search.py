import os
import chromadb

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()


client = chromadb.CloudClient(
    api_key='ck-Fx3PHuVGmyASkJ7cwZBwTcswwxnpvH6TPWLTa1d52UAE',
    tenant='0a9dec27-3647-4676-8611-8de5eb3a8c79',
    database='prototype'
)



def hybridSearch(query):




    # Embedding model
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )


    # Chroma vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        client=client,
        collection_name="prototype_rag"
    )


    # Semantic retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )
    docs = retriever.invoke(query)

    # Get documents for BM25 indexing
    documents = vectorstore.get()["documents"]

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(
        documents
    )

    bm25_retriever.k = 2


    # Combine both retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            retriever,
            bm25_retriever
        ],
        weights=[
            0.5,  # semantic search weight
            0.5   # keyword search weight
        ]
    )


    # Hybrid search
    results = ensemble_retriever.invoke(query)


    return results



if __name__ == "__main__":

    query = "get method in sql"

    results = hybridSearch(query)

    for doc in results:
        print("----------------")
        print(doc.page_content)