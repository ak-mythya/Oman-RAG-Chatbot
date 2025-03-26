import logging
from ...config import llama_llm
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from ...chat_history_manager import ChatHistoryManager

class Generation:
    """
    Uses an LLM to generate the final answer for each sub-query or the original query using aggregated document context and chat history.
    """

    def __init__(self, prompt_path: str = "system_prompts/in-scope.txt"):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "in-scope.txt"
        self.chat_history_manager = ChatHistoryManager() 

    def run(self, state: dict) -> dict:
        logging.info("Running generation node.")
        session_id = state.get("keys", {}).get("session_id", "")
        if not session_id:
            logging.error("No session ID found.")
            return state

    
        chat_history = self.chat_history_manager.get_chat_history(session_id)  # Get chat history for session
        chat_history_text = "\n".join([msg["content"] for msg in chat_history.get("recent_messages", [])])  # Prepare chat history as text

        
        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
        
        generated_responses = []  # To hold all generated responses
        
        # Process each sub-query for generation
        for sq_data in classified_sub_queries:
            sq_text = sq_data.get("completed_query", "")
            classification = sq_data.get("classification", "out-of-scope")
            documents = sq_data.get("documents", [])
            # chat_history = state.get("keys", {}).get("chat_history", "")
            
            # Only generate response for in-scope sub-queries
            if classification == "in-scope":
                logging.info(f"Generating response for sub-query: {sq_text}")
                
                context = "\n\n".join([d.page_content for d in documents])
                
                # Create prompt using in-scope prompt template
                try:
                    with open(self.prompt_path, "r") as f:
                        inscope_prompt = f.read()
                except Exception as e:
                    logging.error(f"Error reading in-scope prompt: {e}")
                    inscope_prompt = ""

                prompt_tmpl = PromptTemplate(
                    template=inscope_prompt,
                    input_variables=["chat_history", "context", "user_query"]
                )
                prompt_str = prompt_tmpl.format(chat_history=chat_history_text, context=context, user_query=sq_text)

                # Generate response for the sub-query
                try:
                    result = llama_llm.invoke(([("system", prompt_str)]))
                    generation = result.content
                except Exception as e:
                    logging.error(f"Error during generation for sub-query: {e}")
                    generation = "Error while generating the response."

                # Append generated response to the sub-query
                sq_data["generated_response"] = generation
                if generation:
                    generated_responses.append({
                        "completed_query": sq_text,
                        "response": generation,
                        "documents": documents
                    })

        # If no sub-queries, process the original query instead
        if not classified_sub_queries:
            original_query = state.get("keys", {}).get("question", "")
            documents = state.get("keys", {}).get("documents", [])
            logging.info(f"Generating response for the original query: {original_query}")

            context = "\n\n".join([d.page_content for d in documents])
            
            # Create prompt for original query
            try:
                with open(self.prompt_path, "r") as f:
                    inscope_prompt = f.read()
            except Exception as e:
                logging.error(f"Error reading in-scope prompt: {e}")
                inscope_prompt = ""

            prompt_tmpl = PromptTemplate(
                template=inscope_prompt,
                input_variables=["chat_history", "context", "user_query"]
            )
            prompt_str = prompt_tmpl.format(chat_history=chat_history_text, context=context, user_query=original_query)

            # Generate response for the original query
            try:
                result = llama_llm.invoke(([("system", prompt_str)]))
                generation = result.content
            except Exception as e:
                logging.error(f"Error during generation for original query: {e}")
                generation = "Error while generating the response."

            generated_responses.append({
                "completed_query": original_query,
                "response": generation,
                "documents": documents
            })

            logging.info(generated_responses)

        # Only update the state with sub-query answers if there are any generated responses
        if generated_responses:
            # Perform the state update once, after collecting all responses
            state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
            state["keys"]["sub_query_mapping"]["sub_query_answers"] = generated_responses
            logging.info(state)
            logging.info(f"Updated sub_query_answers with {len(generated_responses)} responses.")
        else:
            logging.info("No sub-queries generated responses, skipping state update.")

        return state
