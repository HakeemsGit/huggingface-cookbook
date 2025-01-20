import os
import getpass
from typing import Dict, Any
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_sdk import get_client

# Set up environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# Define arithmetic functions
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divides a by b.
    
    Args:
        a: first int
        b: second int
    """
    return a / b

# Set up tools and LLM
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)

# Define system message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Define nodes
def assistant(state: MessagesState) -> Dict[str, Any]:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def human_feedback(state: MessagesState):
    pass

# Build graph
def create_graph(with_feedback: bool = False):
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    if with_feedback:
        builder.add_node("human_feedback", human_feedback)
        builder.add_edge(START, "human_feedback")
        builder.add_edge("human_feedback", "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition
        )
        builder.add_edge("tools", "human_feedback")
        interrupt_node = "human_feedback"
    else:
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition
        )
        builder.add_edge("tools", "assistant")
        interrupt_node = "assistant"
    
    memory = MemorySaver()
    return builder.compile(interrupt_before=[interrupt_node], checkpointer=memory)

# Example usage for basic graph
def run_basic_graph():
    graph = create_graph(with_feedback=False)
    initial_input = {"messages": "Multiply 2 and 3"}
    thread = {"configurable": {"thread_id": "1"}}
    
    print("\nRunning basic graph...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        messages = event.get('messages', [])
        if messages:
            print(f"Message: {messages[-1]}")

# Example usage for graph with feedback
def run_feedback_graph():
    graph = create_graph(with_feedback=True)
    initial_input = {"messages": "Multiply 2 and 3"}
    thread = {"configurable": {"thread_id": "5"}}
    
    print("\nRunning feedback graph...")
    # First run
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        messages = event.get('messages', [])
        if messages:
            print(f"Message: {messages[-1]}")
    
    # Update state with feedback
    user_input = input("Tell me how you want to update the state: ")
    graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")
    
    # Continue execution
    for event in graph.stream(None, thread, stream_mode="values"):
        messages = event.get('messages', [])
        if messages:
            print(f"Message: {messages[-1]}")

# Example usage with LangGraph Studio client
async def run_with_studio_client():
    client = get_client(url="http://localhost:56091")
    
    initial_input = {"messages": "Multiply 2 and 3"}
    thread = await client.threads.create()
    
    print("\nRunning with LangGraph Studio client...")
    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent",
        input=initial_input,
        stream_mode="values",
        interrupt_before=["assistant"],
    ):
        print(f"Event type: {chunk.event}")
        messages = chunk.data.get('messages', [])
        if messages:
            print(f"Message: {messages[-1]}")

if __name__ == "__main__":
    run_basic_graph()
    run_feedback_graph()
    # Note: Async example requires running in an async context
    # import asyncio
    # asyncio.run(run_with_studio_client())