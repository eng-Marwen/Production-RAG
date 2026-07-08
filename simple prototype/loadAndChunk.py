from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os 
def LoadAndChunk(path):
    loader=PyPDFLoader(path)
    documents=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
