from dotenv import load_dotenv  # load .env file into environment variables
import os  # access environment variables and file operations
import tempfile  # create temporary files/directories for processing
from pathlib import Path  # convenient filesystem path utilities (Path objects)
from langchain_community.document_loaders import (
    TextLoader ,
    PyPDFLoader
)
load_dotenv()  # populate os.environ from a .env file in project root

#-----------------------TextLoader Example-----------------------
def load_text_file(file_path: str) -> str:
    #create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write("Hello, this is a test document for the TextLoader.\nIt contains multiple lines of text.\nThis is the third line.")
        tmp_file_path = tmp_file.name
        
    loader = TextLoader(tmp_file_path)
    documents = loader.load()
    print(documents)
    for doc in documents:
        print(f"Loaded document content:\n{doc.page_content}")
    os.remove(tmp_file_path)  # clean up temporary file


#---------------------PyPDFLoader Example-----------------------
def load_pdf_file(file_path: str) -> str:
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    print(documents)
    for doc in documents:   
        print(f"Loaded document content:\n{doc.page_content}")

if __name__ == "__main__":
    #load_text_file("dummy_path.txt")  # path is not used since we create a temp file
    load_pdf_file("./docs/sample.pdf") 