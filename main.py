from dotenv import load_dotenv
from importlib.metadata import version
from langchain_groq import ChatGroq
import os

load_dotenv()

lg_version = version("langgraph")
core_version = version("langchain")
print(f"LangGraph Version: {lg_version}")
print(f"LangChain Core Version: {core_version}")

def main():
    #test groq llm
    llm=ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=500,
    )    
    prompt = "difference between c++ and python"
    response = llm.invoke(prompt)
    print("Response from Groq LLM:")
    print(response.content)
    print("Tokens used:", response.response_metadata["token_usage"]["total_tokens"])


if __name__ == "__main__" :
    main()
