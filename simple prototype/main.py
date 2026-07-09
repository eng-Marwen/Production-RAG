from dotenv import load_dotenv
from importlib.metadata import version
import os
from loadAndChunk import LoadAndChunk
from embeddings import chunksToEmbeddings, ragResponse
# from embeddings import ragResponse
load_dotenv()
if __name__ == "__main__":
    # Load and chunk the documents
    # chunks = LoadAndChunk("./Document 8.pdf")
    # # Create embeddings for the chunks and save them to a vector database
    # embeddings = chunksToEmbeddings(chunks)
    # print(embeddings)
    response=ragResponse("give me the main sql syntax")
    print(response)
