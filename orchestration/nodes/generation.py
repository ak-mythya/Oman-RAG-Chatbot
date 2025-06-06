# import logging
# import asyncio
# import time
# import json
# from model import TextGenerator
# from pathlib import Path
# from langchain_core.prompts import PromptTemplate
# from chat_history_manager import ChatHistoryManager
# import re

# class Generation:
#     """
#     Uses an LLM to generate the final answer for each sub-query or the original query using aggregated document context and chat history.
#     Processes sub-queries in a batch using generate_batch for efficiency, and provides a fallback apology for unanswerable sub-queries.
#     """

#     def __init__(self, prompt_path: str = "system_prompts/in-scope.txt"):
#         current_dir = Path(__file__).resolve().parent
#         self.prompt_path = current_dir.parent.parent / "system_prompts" / "in-scope.txt"
#         self.chat_history_manager = ChatHistoryManager()
#         self.generator = TextGenerator()

#     async def run(self, state: dict) -> dict:
#         logging.info("Running generation node (async).")
#         session_id = state.get("keys", {}).get("session_id", "")
#         if not session_id:
#             logging.error("No session ID found.")
#             return state

#         # Prepare shared data
#         chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id)

#         sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
#         classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

#         # Load the in-scope prompt template
#         try:
#             with open(self.prompt_path, "r", encoding="utf-8") as f:
#                 inscope_prompt = f.read()
#         except Exception as e:
#             logging.error(f"Error reading in-scope prompt: {e}")
#             inscope_prompt = ""

#         prompt_tmpl = PromptTemplate(
#             template=inscope_prompt,
#             input_variables=["chat_history", "context", "user_query"]
#         )

#         generated_responses = []

#         def extract_json_or_return(text: str) -> str:
#             # Regex to find the first JSON object in the text
#             match = re.search(r"(\{.*\})", text, re.DOTALL)
#             if match:
#                 return match.group(1)
#             return text

#         if classified_sub_queries:
#             # Partition into answerable vs unanswerable
#             answerable_sqs   = []
#             unanswerable_sqs = []
#             for sq in classified_sub_queries:
#                 if sq.get("classification") != "in-scope":
#                     continue
#                 if sq.get("answerable", False):
#                     answerable_sqs.append(sq)
#                 else:
#                     unanswerable_sqs.append(sq)

#             # 1) Batch-generate for answerable sub-queries
#             if answerable_sqs:
#                 prompts = []
#                 for sq in answerable_sqs:
#                     context = "\n\n".join(
#                         d.get("page_content") if isinstance(d, dict) else d.page_content
#                         for d in sq.get("documents", [])
#                     )
#                     prompts.append(
#                         prompt_tmpl.format(
#                             chat_history=chat_history_text,
#                             context=context,
#                             user_query=sq["completed_query"]
#                         )
#                     )

#                 logging.info(f"generation_prompt={prompts}")

#                 start = time.time()
#                 logging.info(f"[Gen] START batch generation of {len(prompts)} prompts")
#                 try:
#                     generations = await asyncio.to_thread(
#                         lambda: self.generator.generate_batch(prompts, task="generation")
#                     )
#                 except Exception as e:
#                     logging.error(f"Error during batch generation: {e}")
#                     generations = ["Error while generating the response."] * len(prompts)
#                 end = time.time()
#                 logging.info(f"[Gen] END batch generation at {end:.3f} (took {end-start:.3f}s)")

#                 for sq, gen in zip(answerable_sqs, generations):
#                     generated_responses.append({
#                         "completed_query": sq["completed_query"],
#                         "response": gen,
#                         "documents": sq.get("documents", [])
#                     })

#             # 2) Apologize for unanswerable sub-queries
#             for sq in unanswerable_sqs:
#                 apology = "عذرًا، لا أستطيع الإجابة على هذا السؤال في الوقت الحالي."
#                 generated_responses.append({
#                     "completed_query": sq["completed_query"],
#                     "response": apology,
#                     "documents": sq.get("documents", [])
#                 })

#         else:
#             # Fallback for original query
#             original_query = state.get("keys", {}).get("question", "")
#             documents = state.get("keys", {}).get("documents", [])
#             context = "\n\n".join(
#                 d.get("page_content") if isinstance(d, dict) else d.page_content
#                 for d in documents
#             )
#             prompt_str = prompt_tmpl.format(
#                 chat_history=chat_history_text,
#                 context=context,
#                 user_query=original_query
#             )

#             logging.info(f"generation_prompt={prompt_str}")

#             start = time.time()
#             logging.info(f"[Gen] START original query generation at {start:.3f}")
#             try:
#                 generations = await asyncio.to_thread(
#                     lambda: self.generator.generate_batch([prompt_str], task="generation")
#                 )
#                 generation = generations[0]
#             except Exception as e:
#                 logging.error(f"Error during original query generation: {e}")
#                 generation = "Error while generating the response."
#             end = time.time()
#             logging.info(f"[Gen] END original query generation at {end:.3f} (took {end-start:.3f}s)")

#             # Attempt to extract JSON, else keep original
#             response_text = extract_json_or_return(generation)
#             generated_responses.append({
#                 "completed_query": original_query,
#                 "response": response_text,
#                 "documents": documents
#             })

#         # Update state with all assembled responses
#         if generated_responses:
#             state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
#             state["keys"]["sub_query_mapping"]["sub_query_answers"] = generated_responses
#             logging.info(f"[Gen] Updated sub_query_answers with {len(generated_responses)} responses.")
#             # ─── NEW: if there's exactly one in-scope answer, expose it for the UI ───
#         if len(generated_responses) == 1:
#             state["keys"]["response"]       = generated_responses[0]["response"]
#             state["keys"]["classification"] = "in-scope"
#         logging.info(
#             "Generation Node State Output:\n%s",
#             json.dumps(state, indent=2, ensure_ascii=False)
#         )
#         return state

import logging
import asyncio
import time
import json
from model import TextGenerator
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from chat_history_manager import ChatHistoryManager
import re

def contains_chinese(text: str) -> bool:
    for char in text:
        cp = ord(char)
        if (
            0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or
            0x2B820 <= cp <= 0x2CEAF or
            0x2CEB0 <= cp <= 0x2EBEF or
            0x30000 <= cp <= 0x3134F
        ):
            return True
    return False

def extract_translation_json(text: str) -> str:
    try:
        # match = re.search(r'{\s*"translation"\s*:\s*"(.?)"\s}', text, re.DOTALL)
        match = re.search(r'"translation"\s*:\s*"([^"]*)"', text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        logging.error(f"Regex extraction failed: {e}")
    return text

class Generation:
    def __init__(self, prompt_path: str = "system_prompts/in-scope.txt"):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "in-scope.txt"
        self.chat_history_manager = ChatHistoryManager()
        self.generator = TextGenerator()

    async def run(self, state: dict) -> dict:
        logging.info("Running generation node (async).")
        session_id = state.get("keys", {}).get("session_id", "")
        if not session_id:
            logging.error("No session ID found.")
            return state

        chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id)

        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                inscope_prompt = f.read()
        except Exception as e:
            logging.error(f"Error reading in-scope prompt: {e}")
            inscope_prompt = ""

        prompt_tmpl = PromptTemplate(
            template=inscope_prompt,
            input_variables=["chat_history", "context", "user_query"]
        )

        generated_responses = []

        def extract_json_or_return(text: str) -> str:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return match.group(1)
            return text

        if classified_sub_queries:
            answerable_sqs = []
            unanswerable_sqs = []
            for sq in classified_sub_queries:
                if sq.get("classification") != "in-scope":
                    continue
                if sq.get("answerable", False):
                    answerable_sqs.append(sq)
                else:
                    unanswerable_sqs.append(sq)

            if answerable_sqs:
                prompts = []
                for sq in answerable_sqs:
                    context = "\n\n".join(
                        d.get("page_content") if isinstance(d, dict) else d.page_content
                        for d in sq.get("documents", [])
                    )
                    prompts.append(
                        prompt_tmpl.format(
                            chat_history=chat_history_text,
                            context=context,
                            user_query=sq["completed_query"]
                        )
                    )

                logging.info(f"generation_prompt={prompts}")

                start = time.time()
                logging.info(f"[Gen] START batch generation of {len(prompts)} prompts")
                try:
                    generations = await asyncio.to_thread(
                        lambda: self.generator.generate_batch(prompts, task="generation")
                    )
                except Exception as e:
                    logging.error(f"Error during batch generation: {e}")
                    generations = ["Error while generating the response."] * len(prompts)
                end = time.time()
                logging.info(f"[Gen] END batch generation at {end:.3f} (took {end-start:.3f}s)")

                for sq, gen in zip(answerable_sqs, generations):
                    if contains_chinese(gen):
                        try:
                            translation_prompt = f"""Translate the following Chinese text to Arabic.
                            Retain all the information and meaning.
                            Rules: 
                                1. Only use JSON data types: object, array, string, number, boolean, null.
                                2. Always wrap object keys and string values in double quotes (`\"`).
                                3. Never include comments, explanations, or trailing commas.
                                4. Escape special characters inside strings: "
                                - Newline → `\n`
                                - Tab     → `\t`
                                - Backslash → `\\`
                                - Double‐quote → `\"`
                                5. Do not output any control characters (e.g. unescaped `\r`).
                                *Always output only the JSON structure
                            "with no extra explanation or text.\n\n"  
                            
                            "Here is the text that you have to translate" +
                            {gen}

                            Return only valid JSON in the following format: {{\"translation\": \"<Arabic translation>\"}}
                            """
                            logging.info(f"translation_prompt={translation_prompt}")
                            translated = self.generator.generate(translation_prompt, task="translation")
                            logging.info(f"translated={translated}")
                            if isinstance(translated, list):
                                translated = translated[0]
                            response_text = extract_translation_json(translated)
                            logging.info(f"response_text_translated={response_text}")
                        except Exception as e:
                            logging.error(f"Error during Chinese-to-Arabic translation: {e}")
                            response_text = "تعذر ترجمة المحتوى من الصينية إلى العربية."
                    else:
                        response_text = gen

                    generated_responses.append({
                        "completed_query": sq["completed_query"],
                        "response": response_text,
                        "documents": sq.get("documents", [])
                    })

            for sq in unanswerable_sqs:
                apology = "عذرًا، لا أستطيع الإجابة على هذا السؤال في الوقت الحالي."
                generated_responses.append({
                    "completed_query": sq["completed_query"],
                    "response": apology,
                    "documents": sq.get("documents", [])
                })

        else:
            original_query = state.get("keys", {}).get("question", "")
            documents = state.get("keys", {}).get("documents", [])
            context = "\n\n".join(
                d.get("page_content") if isinstance(d, dict) else d.page_content
                for d in documents
            )
            prompt_str = prompt_tmpl.format(
                chat_history=chat_history_text,
                context=context,
                user_query=original_query
            )

            logging.info(f"generation_prompt={prompt_str}")

            start = time.time()
            logging.info(f"[Gen] START original query generation at {start:.3f}")
            try:
                generations = await asyncio.to_thread(
                    lambda: self.generator.generate_batch([prompt_str], task="generation")
                )
                generation = generations[0]
            except Exception as e:
                logging.error(f"Error during original query generation: {e}")
                generation = "Error while generating the response."
            end = time.time()
            logging.info(f"[Gen] END original query generation at {end:.3f} (took {end-start:.3f}s)")

            response_text = extract_json_or_return(generation)
            if contains_chinese(response_text):
                try:
                    translation_prompt = (
                                "Translate the following Chinese text to Arabic. "
                                "Retain all the information and meaning. "
                                "Return only valid JSON in the following format: {\"translation\": \"<Arabic translation>\"} "
                                "Rules: "
                                "1. Only use JSON data types: object, array, string, number, boolean, null. "
                                "2. Always wrap object keys and string values in double quotes (`\"`). "
                                "3. Never include comments, explanations, or trailing commas. "
                                "4. Escape special characters inside strings: "
                                "- Newline → `\n` "
                                "- Tab     → `\t` "
                                "- Backslash → `\\` "
                                "- Double‐quote → `\"` "
                                "5. Do not output any control characters (e.g. unescaped `\r`). "
                                "*Always output only the JSON structure"
                                "with no extra explanation or text.\n\n" + gen
                            )
                    translated = self.generator.generate(translation_prompt, task="translation")
                    if isinstance(translated, list):
                        translated = translated[0]
                    response_text = extract_translation_json(translated)
                except Exception as e:
                    logging.error(f"Error during Chinese-to-Arabic fallback translation: {e}")
                    response_text = "تعذر ترجمة المحتوى من الصينية إلى العربية."

            generated_responses.append({
                "completed_query": original_query,
                "response": response_text,
                "documents": documents
            })

        if generated_responses:
            state.setdefault("keys", {}).setdefault("sub_query_mapping", {})
            state["keys"]["sub_query_mapping"]["sub_query_answers"] = generated_responses
            logging.info(f"[Gen] Updated sub_query_answers with {len(generated_responses)} responses.")

        if len(generated_responses) == 1:
            state["keys"]["response"] = generated_responses[0]["response"]
            state["keys"]["classification"] = "in-scope"

        logging.info("Generation Node State Output:\n%s", json.dumps(state, indent=2, ensure_ascii=False))
        return state