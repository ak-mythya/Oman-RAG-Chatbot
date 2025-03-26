import json
import logging
from pathlib import Path
from ...config import llama_llm
from ...chat_history_manager import ChatHistoryManager 

class FinalResponseGenerator:
    """
    Generates the final, personalized response by integrating sub-query responses,
    chat history, and the original user query.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "final_response.txt"
        self.chat_history_manager = ChatHistoryManager()

    def run(self, state: dict) -> dict:
        logging.info("Running final response generation node.")
        user_query = state.get("keys", {}).get("question", "").strip()
      
        session_id = state.get("keys", {}).get("session_id", "")
        if not session_id:
            logging.error("No session ID found.")
            return state

        # Retrieve chat history for the session
        chat_history = self.chat_history_manager.get_chat_history(session_id)

        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        identified_sub_queries = sub_query_mapping.get("sub_queries", [])
        sub_query_answers = sub_query_mapping.get("sub_query_answers", [])

        # Construct sub-query context parts based on answers or identified queries
        sub_query_context_parts = []
        if sub_query_answers:
            for idx, ans in enumerate(sub_query_answers, start=1):
                completed_query = ans.get("completed_query", "").strip()
                response = ans.get("response", "").strip()
                if completed_query or response:
                    sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
        elif identified_sub_queries:
            for idx, sq in enumerate(identified_sub_queries, start=1):
                completed_query = sq.get("completed_query", "").strip() if isinstance(sq, dict) else str(sq).strip()
                response = sq.get("response", "").strip() if isinstance(sq, dict) else ""
                if completed_query or response:
                    sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
        
        sub_query_context = "\n\n".join(sub_query_context_parts)

        # Generate the final prompt for the LLM to generate a response
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                final_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading final response prompt: {e}")
            final_prompt = ""

        prompt = final_prompt.format(
            chat_history="\n".join([msg["content"] for msg in chat_history["recent_messages"]]),
            sub_query_context=sub_query_context,
            user_query=user_query
        )

        try:
            final_response = llama_llm.invoke(([("system", prompt)]))
            final_response_text = final_response.content
        except Exception as e:
            logging.error(f"Error generating final response: {e}")
            final_response_text = "Error while generating the final response."

        # Retrieve documents if available
        documents = state.get("keys", {}).get("documents", [])
        
        # Save final response generation details in the state
        state.setdefault("keys", {})["final_response_generation"] = {
            "user_query": user_query,
            "final_response": final_response_text,
            "documents": documents
        }
         
        user_message = user_query
        assistant_message = final_response_text
        self.chat_history_manager.update_chat_history(session_id, user_message, assistant_message)

        return state
