import logging
from ...config import llama_llm
from pathlib import Path
from langchain_core.prompts import PromptTemplate

class Generation:
    """
    Uses an LLM to generate the final answer using aggregated document context and chat history.
    """

    def __init__(self, prompt_path: str = "system_prompts/in-scope.txt"):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "in-scope.txt"

    def run(self, state: dict) -> dict:
        logging.info("Running generation node.")
        question = state.get("keys", {}).get("question", "")
        documents = state.get("keys", {}).get("documents", [])
        chat_history = state.get("keys", {}).get("chat_history", "")

        context = "\n\n".join([d.page_content for d in documents])
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
        prompt_str = prompt_tmpl.format(chat_history=chat_history, context=context, user_query=question)
        try:
            result = llama_llm.invoke(([("system", prompt_str)]))
            generation = result.content
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            generation = "Error while generating the response."

        # Update sub_query_mapping in the state.
        state.setdefault("keys", {}).setdefault("sub_query_mapping", {
            "original_query": question,
            "sub_queries": [],
            "sub_query_answers": []
        })
        state["keys"]["sub_query_mapping"]["sub_query_answers"].append({
            "completed_query": question,
            "response": generation,
            "documents": documents
        })
        state["keys"]["generated_answer"] = generation
        return state
