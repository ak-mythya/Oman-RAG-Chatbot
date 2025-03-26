"""
Entry point for the advanced RAG chatbot.
"""

from .orchestration.graph_assembly import app

def run_advanced_rag_pipeline(question: str, session_id: str) -> str:
    inputs = {"keys": {"question": question, "session_id": session_id}}
    final_answer = "No final generation produced."
    
    for output in app.stream(inputs):
        for step_name, step_data in output.items():
            final_answer = step_data["keys"].get("final_response_generation", final_answer)
    
    return final_answer

def main():
    user_question = "Who is Modi?"
    answer = run_advanced_rag_pipeline(user_question, "125")
    print("Final Answer:\n", answer)

if __name__ == "__main__":
    main()
