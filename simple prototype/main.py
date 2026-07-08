from dotenv import load_dotenv
from importlib.metadata import version
import os
from loadAndChunk import LoadAndChunk
from embeddings import chunksToEmbeddings
load_dotenv()
if __name__ == "__main__":
    # Load and chunk the documents
    chunks = LoadAndChunk("./Document 8.pdf")
    print(chunks[0])
    # Create embeddings for the chunks
    #embeddings = chunksToEmbeddings(chunks)

