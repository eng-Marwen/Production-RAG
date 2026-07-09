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

"""
==========================================================
LangChain ChatGroq Example
==========================================================

Purpose:
- Uses LangChain's ChatGroq wrapper to access Groq models.
- Designed for building LLM applications such as RAG,
  AI agents, chatbots, and multi-step workflows.

How it works:
1. Load the API key from the environment.
2. Create a ChatGroq LLM object.
3. Send a prompt using llm.invoke().
4. Receive an AIMessage object containing the response.

Advantages:
- Unified interface for different LLM providers.
- Easily integrates with Prompt Templates, Document Loaders,
  Retrievers, Vector Stores, Memory, Chains, and LangGraph.
- Easy to switch between Groq, OpenAI, Anthropic, Gemini, etc.
- Recommended for scalable AI applications.

Use it when you want to build a complete AI application and Agents
Limitations:
- Requires LangChain dependencies.
- Adds a small abstraction layer over the native API.
"""

if __name__ == "__main__" :
    main()
