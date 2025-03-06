from dotenv import load_dotenv
import functools
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from helpers import agent_node,create_agent
from tools import LeadFinderTool, LeadExtractorTool
from typing import Type, List, Dict, Any
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

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
graph.add_node("lead_finder", lead_finder_node)
graph.add_node("lead_enricher", lead_enricher_node)
graph.add_node("call_tool", ToolNode(lead_finder_tools + lead_enricher_tools))

# Edges
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
    {"lead_finder": "lead_finder", "lead_enricher": "lead_enricher"},
)

graph.add_edge(START, "lead_finder")

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