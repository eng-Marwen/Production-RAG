
import os
from dotenv import load_dotenv
load_dotenv()

#enable tracing for LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.run_trees import RunTree

@traceable(name="demo_basic_tracing", project=os.getenv("LANGSMITH_PROJECT"))
def demo_basic_tracing():
    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "explain well this topic {topic} in 2 phrases?"
    )
    chain= prompt | llm  #inejct the prompt into the llm
    response= chain.invoke({"topic": "cloud computing"}) # run the chain with the topic 
    print("Response:", response.content)

@traceable(name="named_runs_demo", tags=["production","summarization"], project="second-tracing")
def demo_tags_tracing():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    response = llm.invoke("Explain well cloud computing in 2 phrases?")
    print(response.content)
    print("Response:", response.content)

@traceable(name="metadata", tags=["metadata","filtering"], project="hhhhhh")
def demo_metadata_tracing(user_id:int, request_type:str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    response = llm.invoke("Explain well edge computing in 2 phrases?")
    print(response.content)
    return response.content

if __name__ == "__main__":
    demo_basic_tracing()
    demo_tags_tracing()
    demo_metadata_tracing(user_id=123,request_type="greeting")