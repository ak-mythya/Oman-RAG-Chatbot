from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from .nodes.sub_query_identification import SubQueryIdentifier
from .nodes.query_classification import QueryClassifier
from .nodes.retrieval_node import ContextAwareRetriever
from .nodes.document_grader_node import DocumentGrader
from .nodes.query_transformer_node import QueryTransformer
from .nodes.web_search_node import WebSearch
from .nodes.clarifying_question import Clarifying_Question
from .nodes.generation import Generation
from .nodes.final_response_generation import FinalResponseGenerator
from .nodes.general_out_of_scope import GeneralQueryNode
from .nodes.sub_query_loop import SubQueryLoop

# Define GraphState type
class GraphState(TypedDict):
    keys: Dict[str, any]

# Instantiate workflow
workflow = StateGraph(GraphState)

# Instantiate nodes
identify = SubQueryIdentifier().run
sub_query_loop = SubQueryLoop().run
generate = Generation().run
finalize = FinalResponseGenerator().run
general = GeneralQueryNode().run

# Clarification-path nodes
clarify = Clarifying_Question().run
retrieve = ContextAwareRetriever().run
grade = DocumentGrader().run
transform = QueryTransformer().run
# reuse classify if you want to reclassify after transform
reclassify = QueryClassifier().run

document_grader = DocumentGrader()  # for threshold reference

# human_feedback uses interrupt
async def human_feedback(state: GraphState) -> GraphState:
    keys = state.setdefault("keys", {})
    question = keys.get("clarifying_question", "Could you clarify?")
    reply: str = interrupt(question)
    keys["user_feedback"] = reply.strip()
    return state


async def early_escalate(state: GraphState) -> GraphState:
    print("Running early escalation node ...")
    keys = state["keys"]
    classes = keys.get("escalation_classes", [])
    # define your precedence:
    for cls in ("exit-chat", "human-agent", "service-application", "out-of-scope"):
        if cls in classes:
            sel = cls
            break
    else:
        return state 

    # now emit the right UX for sel
    if sel == "exit-chat":
        keys["escalation_message"] = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
        keys["show_buttons"]     = True
        keys["button_options"]   = ["الاستمرار", "البدء من جديد"]
    elif sel == "human-agent":
        keys["escalation_message"] = "هل تريد الاتصال بوكيل بشري؟"
        keys["show_buttons"]       = True
        keys["button_options"]     = [
            "نعم، أريد التواصل مع عميل بشري",
            "لا، تابع"
        ]
    elif sel == "service-application":
        keys["escalation_message"] = "يرجى تقديم رقم الهوية المدنية."
        keys["show_buttons"]       = False

    elif sel == "child-abuse":
        keys["escalation_message"] = "يرجى العثور على رابط نموذج إساءة معاملة الأطفال: https://portal.mosd.gov.om/webcenter/portal/MOSDExternalPortal/pages_services/reportabuse"
        keys["show_buttons"]       = False
       
    else:  # out-of-scope
        keys["escalation_message"] = (
            "نعتذر، الموضوع خارج اختصاص عمل الوزارة"
        )
        keys["show_buttons"]       = False

    keys["classification"] = sel
    return state

# Register all nodes
workflow.add_node("identify_sub_queries", identify)
workflow.add_node("sub_query_loop", sub_query_loop)
workflow.add_node("early_escalate", early_escalate)
workflow.add_node("general",general)

workflow.add_node("generate", generate)
workflow.add_node("final_response_generation", finalize)
workflow.add_node("clarifying_question", clarify)
workflow.add_node("human_feedback", human_feedback)
workflow.add_node("transform_query", transform)
workflow.add_node("classify_after_clarify", reclassify)
workflow.add_node("retrieve_after_clarify", retrieve)
workflow.add_node("grade_after_clarify", grade)

# Entry point and original flow
workflow.set_entry_point("identify_sub_queries")
workflow.add_edge("identify_sub_queries", "sub_query_loop")

workflow.add_conditional_edges(
    "sub_query_loop",
    lambda s: (
        "final_response_generation" if s["keys"].get("multiple_no_inscope", False)
        else "early_escalate" if s["keys"].get("has_escalation_request", False)
        else "general"      if s["keys"].get("only_general_request", False)
        else "clarifying_question" if s["keys"].get("need_clarification", False)
        else "generate"
    ),
    {
        "final_response_generation": "final_response_generation",
        "early_escalate":      "early_escalate",
        "general":             "general",
        "clarifying_question":"clarifying_question",
        "generate":            "generate",
    }
)

# Clarification‐feedback‐transform‐reclassify‐retrieve‐grade loop
workflow.add_edge("clarifying_question", "human_feedback")
workflow.add_edge("human_feedback", "transform_query")
workflow.add_edge("transform_query", "classify_after_clarify")
workflow.add_edge("classify_after_clarify", "retrieve_after_clarify")
workflow.add_edge("retrieve_after_clarify", "grade_after_clarify")
workflow.add_edge("grade_after_clarify", "generate")

workflow.add_conditional_edges(
    "generate",
    lambda s: (
        "END" if s["keys"].get("only_one_in_scope", False)
        else "final_response_generation"
    ),
    {
        "END": END,
        "final_response_generation": "final_response_generation",
    }
)

workflow.add_edge("early_escalate", END)

workflow.add_edge("general", END)

# Compile the graph
checkpointer = MemorySaver()
# Compile
# app = workflow.compile(checkpointer=MemorySaver())
app = workflow.compile(checkpointer=checkpointer)