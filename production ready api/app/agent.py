from typing import Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from app.config import get_settings


class AgentState(TypedDict):
    """
    State for the production agent.
    uses annotetd with add_messages reducer for message accumulation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    used_model: str


class ProductionAgent(StateGraph[AgentState]):
    """
    Production langgraph agent:
    -retrty on failure (fallback model)
    -Graceful error handling
    -langsmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatGroq(
            model=settings.primary_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=0,  # we handle retries ourselves
            max_tokens=300,
        )

        self.fallback_llm = ChatGroq(
            model=settings.fallback_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=0,  # we handle retries ourselves
        )

        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the langgraph state machine.
        """

        # NODES
        def process_message(state: AgentState) -> dict:
            """Process the incoming message using the primary LLM."""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {"messages": [response], "used_model": "primary", "error": None}
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "used_model": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Try the fallback LLM if the primary fails."""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {"messages": [response], "used_model": "fallback", "error": None}
            except Exception as e:
                return {"error": str(e), "used_model": ""}

        def handle_error(state: AgentState) -> dict:
            """Handle errors gracefully."""
            return {
                "messages": [
                    AIMessage(
                        content=" I'm sorry, but I encountered an error while processing your request. Please try again later."
                    )
                ],
                "used_model": "error_handler",
            }

        # EDGES
        def route_after_process(state: AgentState) -> str:
            """decide what to do after primary attempt"""
            if state["error"] is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "try_fallback"
            else:
                return "handle_error"

        def route_after_fallback(state: AgentState) -> str:
            """decide what to do after fallback attempt"""
            if state["error"] is None:
                return "done"
            else:
                return "error"

        # Build the state graph
        graph = StateGraph(AgentState)

        graph.add_node("process_message", process_message)
        graph.add_node("try_fallback", try_fallback)
        graph.add_node("handle_error", handle_error)

        graph.add_edge(START, "process_message")

        graph.add_conditional_edges(
            "process_message",
            route_after_process,
            {
                "done": END,
                "try_fallback": "try_fallback",
                "handle_error": "handle_error",
            },
        )

        graph.add_conditional_edges(
            "try_fallback", route_after_fallback, {"done": END, "error": "handle_error"}
        )
        graph.add_edge("handle_error", END)
        return graph.compile()

    @traceable(name="testtt", project_name="marwen")
    def invoke(self, messages: str) -> dict:
        """
        Invoke the production agent with the given messages.
        Returns {"response": str, "used_model": str, "error": Optional[str]}
        """
        result = self.graph.invoke(
            input={
                "messages": [HumanMessage(content=messages)],
                "error": None,
                "retry_count": 0,
                "used_model": "",
            }
        )
        return {
            "response": result["messages"][-1].content,
            "used_model": result["used_model"],
            "error": result["error"],
        }
