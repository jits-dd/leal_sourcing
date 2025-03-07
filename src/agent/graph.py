from dotenv import load_dotenv
import functools
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI  # Use LangChain's ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from helpers import agent_node, create_agent
from tools import LeadFinderTool, LeadExtractorTool
from typing import Type, List, Dict, Any

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str

# Initialize Perplexity API using LangChain's ChatOpenAI
llm = ChatOpenAI(
    model="sonar-pro",  # Use the appropriate Perplexity model
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",  # Perplexity API endpoint
)

def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    return "continue"

# Tool definitions
lead_finder_tools = [LeadFinderTool()]
lead_enricher_tools = [LeadExtractorTool()]

graph = StateGraph(State)

# Task Planner
task_planner_agent = create_agent(
    llm,
    [],
    """
    You are a professional task planner. 
    Your job is to break down the user's requirement into sub-tasks that can be executed by the lead finder.
    Return only the list of sub-tasks.
    Do not include any markdown styling in your response.
    """,
)
task_planner_node = functools.partial(
    agent_node, agent=task_planner_agent, name="task_planner"
)

# Lead Finder
lead_finder_agent = create_agent(
    llm,
    lead_finder_tools,
    """
    You are a professional lead finder. 
    Your job is to find companies that are actively acquiring or open to acquisitions in a specific industry and location.
    Return only the list of found companies, along with their details (e.g., name, industry, acquisition history, strategic fit).
    Do not include any markdown styling in your response.
    """,
)
lead_finder_node = functools.partial(
    agent_node, agent=lead_finder_agent, name="lead_finder"
)

# Lead Enricher
lead_enricher_agent = create_agent(
    llm,
    lead_enricher_tools,
    """
    You are a professional lead enricher. Your job is to gather as much relevant information about potential buyers as possible.
    You will receive a list of companies and use the LeadExtractorTool to gather additional data from the provided sources.
    The enriched lead data must contain the following fields: company name, industry, acquisition history, strategic fit, and relevance.
    If any of these fields are not available, mark them as "not found".
    Do not include any markdown styling in your response.
    """,
)
lead_enricher_node = functools.partial(
    agent_node, agent=lead_enricher_agent, name="lead_enricher"
)

# Nodes
graph.add_node("task_planner", task_planner_node)
graph.add_node("lead_finder", lead_finder_node)
graph.add_node("lead_enricher", lead_enricher_node)
graph.add_node("call_tool", ToolNode(lead_finder_tools + lead_enricher_tools))

# Edges
graph.add_conditional_edges(
    "task_planner",
    router,
    {"continue": "lead_finder", "call_tool": "call_tool"},
)
graph.add_conditional_edges(
    "lead_finder",
    router,
    {"continue": "lead_enricher", "call_tool": "call_tool"},
)
graph.add_conditional_edges(
    "lead_enricher", router, {"continue": END, "call_tool": "call_tool"}
)
graph.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"task_planner": "task_planner", "lead_finder": "lead_finder", "lead_enricher": "lead_enricher"},
)

graph.add_edge(START, "task_planner")

graph = graph.compile()
graph.name = "Leadgen Graph"

# Function to collect company details interactively
def collect_company_details():
    print("Welcome to the Lead Sourcing Agent!")
    requirement = input("Enter your requirement ")
    print("\nThank you! Finding potential buyers...\n")
    return {
        "role": "user",
        "content": f"{requirement}",
    }

# Uncomment when you want to run the graph locally. If doing so, check the Firecrawl url in tools.py.

async def main():
    inputs = [collect_company_details()]
    async for chunk in graph.astream({"messages": inputs}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

# Run the script
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())