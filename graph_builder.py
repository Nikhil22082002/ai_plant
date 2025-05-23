# graph_builder.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List # Import TypedDict and List
from nodes import (
    check_math_related_node,
    retrieve_from_knowledge_base_node,
    perform_web_search_node,
    extract_from_web_results_node,
    generate_solution_node,
)

# Define the state for our graph using TypedDict
class AgentState(TypedDict): # Changed from class to TypedDict
    query: str
    kb_result: str
    web_results: List[dict] # Use List from typing
    extracted_content: str
    solution: str
    feedback: str
    error: str

# Create a LangGraph StateGraph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("check_math_related", check_math_related_node)
workflow.add_node("retrieve_from_kb", retrieve_from_knowledge_base_node)
workflow.add_node("perform_web_search", perform_web_search_node)
workflow.add_node("extract_from_web", extract_from_web_results_node)
workflow.add_node("generate_solution", generate_solution_node)

# Set the entry point
workflow.set_entry_point("check_math_related")

# Define the edges (transitions between nodes)
workflow.add_conditional_edges(
    "check_math_related",
    # The lambda function should return a string key that matches one of the edges
    lambda state: "math_related" if not state.get("error") else "not_math_related",
    {
        "math_related": "retrieve_from_kb",
        "not_math_related": END, # End if not math-related
    },
)

workflow.add_conditional_edges(
    "retrieve_from_kb",
    # The lambda function should return a string key that matches one of the edges
    lambda state: "kb_hit" if state["kb_result"] else "kb_miss",
    {
        "kb_hit": "generate_solution", # Go directly to generate_solution if KB hit
        "kb_miss": "perform_web_search",
    },
)

workflow.add_edge("perform_web_search", "extract_from_web")
workflow.add_edge("extract_from_web", "generate_solution")

# The generate_solution node is now the final step before ending the graph
workflow.add_edge("generate_solution", END)

# Compile the graph
app = workflow.compile()

