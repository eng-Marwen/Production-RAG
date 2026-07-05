import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
load_dotenv()  

def chat():
    client=Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
    conversation = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print("Response from Groq SDK:")
    print(conversation.choices[0].message)

def embedding():
    client= OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="Hello, world!"       
    )
    print("Embedding from Groq SDK:")
    print(response.data[0].embedding)

if __name__ == "__main__":
    #chat() 
    embedding()
"""
==========================================================
Native Groq SDK Example
==========================================================

Purpose:
- Uses the official Groq Python SDK to communicate directly
  with the Groq API.
- Suitable for simple chat applications, testing models,
  and lightweight scripts.

How it works:
1. Load the API key from the environment.
2. Create a Groq client.
3. Send a request using chat.completions.create().
4. Receive the model's response.

Advantages:
- Simple and lightweight.
- Full control over the API request.
- Minimal dependencies.
- Good for learning the Groq API.

Limitations:
- Conversation history must be managed manually.
- No built-in support for RAG, memory, agents, or prompt templates.
- More code is required for advanced AI applications.
"""  
