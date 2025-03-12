"""
Advanced Graph Assembly integrating the modular class‑based pipeline nodes.
"""

from typing import TypedDict, Dict
from langgraph.graph import END, StateGraph

# Import class-based nodes.
from .nodes.sub_query_identification import SubQueryIdentifier
from .nodes.query_classification import QueryClassifier
from .nodes.retrieval_node import Retriever
from .nodes.document_grader_node import DocumentGrader
from .nodes.query_transformer_node import QueryTransformer
from .nodes.web_search_node import WebSearch
from .nodes.generation import Generation
from .nodes.final_response_generation import FinalResponseGenerator
from .nodes.general_out_of_scope import GeneralQueryNode, OutOfScopeQueryNode
from .nodes.decide_next_step import DecideNextStep

# Define the graph state type.
class GraphState(TypedDict):
    keys: Dict[str, any]

# Initialize the state graph.
workflow = StateGraph(GraphState)

# Instantiate nodes.
sub_query_identifier = SubQueryIdentifier()
query_classifier = QueryClassifier()
retriever = Retriever()
document_grader = DocumentGrader()
query_transformer = QueryTransformer()
web_search = WebSearch()
generation = Generation()
final_response_generator = FinalResponseGenerator()
handle_general_query = GeneralQueryNode()
handle_out_of_scope_query = OutOfScopeQueryNode()
decide_next_step = DecideNextStep(threshold=3)

# Register nodes in the graph.
workflow.add_node("identify_sub_queries", sub_query_identifier.run)
workflow.add_node("classify_query", query_classifier.run)
workflow.add_node("retrieve", retriever.run)
workflow.add_node("grade_documents", document_grader.run)
workflow.add_node("transform_query", query_transformer.run)
workflow.add_node("web_search", web_search.run)
workflow.add_node("generate", generation.run)
workflow.add_node("final_response_generation", final_response_generator.run)
workflow.add_node("handle_general_query", handle_general_query.run)
workflow.add_node("handle_out_of_scope_query", handle_out_of_scope_query.run)
workflow.add_node("decide_next_step", decide_next_step.run)

# Define graph routing.
workflow.set_entry_point("identify_sub_queries")
workflow.add_edge("identify_sub_queries", "classify_query")
workflow.add_conditional_edges(
    "classify_query",
    lambda state: state.get("keys", {}).get("classification", "out-of-scope"),
    {
        "in-scope": "retrieve",
        "general": "handle_general_query",
        "out-of-scope": "handle_out_of_scope_query",
    },
)
# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_edge("grade_documents", "decide_next_step")
# # After "grade_documents", route to "decide_next_step"

# workflow.add_conditional_edges(
#     "decide_next_step",
#     # This lambda reads the route from the state dictionary
#     lambda st: st["keys"].get("_next_route", "generate"),
#     {
#         "transform_query": "transform_query",
#         "web_search": "web_search",
#         "generate": "generate",
#     },
# )

# # The rest of your edges remain the same
# workflow.add_edge("transform_query", "retrieve")  # Re-run retrieval with transformed query
# workflow.add_edge("web_search", "generate")      # Or do retrieval after web search
# workflow.add_edge("retrieve", "grade_documents")  # Then grade docs again
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "final_response_generation")
workflow.add_edge("handle_general_query", "final_response_generation")
workflow.add_edge("handle_out_of_scope_query", "final_response_generation")
workflow.add_edge("final_response_generation", END)

# Compile the graph.
app = workflow.compile()
