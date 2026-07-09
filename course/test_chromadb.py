import chromadb
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)
# Direct ChromaDB usage (low-level API)
# - You manually manage documents, ids, and metadata
# - Full control over how data is stored and queried
# - Returns raw dictionary results
# - Useful for production-grade vector DB control without abstractions

def chroma_basics():
    collection = client.get_or_create_collection("my_collection")
    add_collection = collection.upsert(
        documents=["Hello world", "This is a test document.","my dog is so cute like a baby"],
        metadatas=[{"source": "test1"}, {"source": "test2"}, {"source": "test3"}],
        ids=["doc1", "doc2", "doc3"]
    )
    print("Collection created and documents added successfully.")

    result=collection.query(
        query_texts=["Hello from the other world"],
        n_results=2
    )
    print("Query result:")
    print(result)
    return client   

# LangChain Chroma usage (high-level abstraction)(Chroma wrapper)
# - Works with Document objects instead of raw strings
# - Automatically handles embeddings and formatting
# - Returns clean (Document, score) results
# - Best suited for RAG systems and LLM applications

def get_similarity_with_scores():
    # create vector store from documents (WRAP STRINGS)
    SAMPLE_DOCS = [
        Document(page_content="Hello world"),
        Document(page_content="This is a test document."),
        Document(page_content="my dog is so cute like a baby"),
    ]

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        client=client,
        collection_name="my_collection",
        embedding=None  # Use default embedding model
    )

    query = "Hello from the other world"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")
    for i, (doc, score) in enumerate(results_with_scores): #distance score
        print(f"Result {i+1}: {doc.page_content} (Score: {score:.4f})")

    return results_with_scores

#in metadata filtering, you can filter the results based on metadata fields. 
# For example, if you want to retrieve documents that have a specific source,topic,age ...
#  you can use a filter criteria.

def metadata_filtering():
    vectorstore = Chroma.from_documents(
        documents=[
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="This is a test document.", metadata={"source": "test2"}),
            Document(page_content="my dog is so cute like a baby", metadata={"source": "test3"}),
        ],
        client=client,
        collection_name="my_collection",
        embedding=None  # Use default embedding model
    )
    filter_creteria = {"source": "test2"}
    results_with_scores = vectorstore.similarity_search(
        query="test document",
        k=3,
        filter=filter_creteria
    )
    print(f"Filtered results for criteria {filter_creteria}:")
    for i, doc in enumerate(results_with_scores):
        print(f"Result {i+1}: {doc.page_content} source: {doc.metadata.get('source')}")

if __name__ == "__main__":
    # chroma_basics()
    # get_similarity_with_scores()
    metadata_filtering()

