import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

client = chromadb.CloudClient(
        api_key='ck-Fx3PHuVGmyASkJ7cwZBwTcswwxnpvH6TPWLTa1d52UAE',
        tenant='0a9dec27-3647-4676-8611-8de5eb3a8c79',
        database='prototype'
)

def chunksToEmbeddings(chunks):

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )
    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )   


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="database_rag",
        collection_metadata={
        "hnsw:space": "cosine",         # Distance metric used for similarity search (cosine, l2, or ip)
        "hnsw:M": 32,                   # Maximum number of neighbor connections per node (higher = better recall, more memory)
        "hnsw:construction_ef": 400,    # Search effort during index construction (higher = better graph quality, slower indexing)
        "hnsw:search_ef": 100,          # Search effort during queries (higher = better recall, slower search)
        }
    )
    return embeddings


def ragResponse(query):
    # Load embedding model
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

    # Connect to Chroma
    vectorstore = Chroma(
        embedding_function=embeddings,
        client=client,
        collection_name="prototype_rag"
    )

    # Retrieve top-2 relevant documents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(query)

    # Convert documents to a single context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant that answers questions using the provided context.

            If the answer cannot be found in the context, respond with:
            "I don't know based on the provided documents."

            Context:
            {context}

            Question:
            {question}""")

    # Create chain
    chain = prompt | llm

    # Invoke model
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content