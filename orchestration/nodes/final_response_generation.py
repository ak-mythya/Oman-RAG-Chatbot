# import json
# import logging
# from pathlib import Path
# from model import TextGenerator
# from chat_history_manager import ChatHistoryManager 
# import re

# class FinalResponseGenerator:
#     """
#     Generates the final, personalized response by integrating sub-query responses,
#     chat history, and the original user query.
#     """

#     def __init__(self):
#         current_dir = Path(__file__).resolve().parent
#         self.prompt_path = current_dir.parent.parent / "system_prompts" / "final_response.txt"
#         self.chat_history_manager = ChatHistoryManager()
#         self.generator = TextGenerator()

#     async def run(self, state: dict) -> dict:
#         logging.info("Running final response generation node.")
#         user_query = state.get("keys", {}).get("question", "").strip()
#         session_id = state.get("keys", {}).get("session_id", "")
#         if not session_id:
#             logging.error("No session ID found.")
#             return state

#         # Retrieve and format chat history for the session
#         chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id, max_messages=5)

#         sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
#         identified_sub_queries = sub_query_mapping.get("sub_queries", [])
#         sub_query_answers = sub_query_mapping.get("sub_query_answers", [])
#         classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

#         # Construct sub-query context parts and collect documents
#         sub_query_context_parts = []
#         all_documents = []
#         if sub_query_answers:
#             for idx, ans in enumerate(sub_query_answers, start=1):
#                 completed_query = ans.get("completed_query", "").strip()
#                 response = ans.get("response", "").strip()
#                 documents = ans.get("documents", [])
#                 if completed_query or response:
#                     sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
#                 all_documents.extend(documents)
#         elif identified_sub_queries:
#             for idx, sq in enumerate(identified_sub_queries, start=1):
#                 completed_query = sq.get("completed_query", "").strip() if isinstance(sq, dict) else str(sq).strip()
#                 response = sq.get("response", "").strip() if isinstance(sq, dict) else ""
#                 documents = sq.get("documents", []) if isinstance(sq, dict) else []
#                 if completed_query or response:
#                     sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
#                 all_documents.extend(documents)

#         # Then, add non-in-scope queries from classified_sub_queries
#         non_inscope_classifications = {"child-abuse", "out-of-scope", "general"}
#         for sq in classified_sub_queries:
#             classification = sq.get("classification", "")
#             if classification in non_inscope_classifications:
#                 completed_query = sq.get("completed_query", "").strip()
#                 response = sq.get("response", "").strip()
#                 documents = sq.get("documents", [])
#                 if completed_query or response:
#                     # Avoid duplicating sub-queries already included via sub_query_answers
#                     if not any(ans.get("completed_query") == completed_query for ans in sub_query_answers):
#                         sub_query_context_parts.append(f"Sub-Query {len(sub_query_context_parts) + 1}: {completed_query}\nResponse {len(sub_query_context_parts) + 1}: {response}")
#                 all_documents.extend(documents)
        
#         sub_query_context = "\n\n".join(sub_query_context_parts)

#         # Check classifications to determine response behavior
#         if classified_sub_queries:
#             classifications = [sq["classification"] for sq in classified_sub_queries]
#             # Handle "exit-chat" or "out-of-scope" cases
#             if "exit-chat" in classifications or all(cls == "out-of-scope" for cls in classifications):
#                 final_response_json = {
#                     "user_query": user_query,
#                     "final_response": "",
#                     "extra_response": "هل تريد مناقشة موضوع آخر؟" if "exit-chat" in classifications else ""
#                 }
#                 final_response_text = json.dumps(final_response_json, ensure_ascii=False)
#             else:
#                 # Generate the final prompt for the LLM
#                 try:
#                     with open(self.prompt_path, "r", encoding="utf-8") as f:
#                         final_prompt = f.read()
#                 except Exception as e:
#                     logging.error(f"Error reading final response prompt: {e}")
#                     final_prompt = ""

#                 prompt = final_prompt.format(
#                     chat_history=chat_history_text,
#                     sub_query_context=sub_query_context,
#                     user_query=user_query
#                 )

#                 logging.info(f"final_response_prompt={prompt}")

#                 try:
#                     final_response_text = self.generator.generate(prompt=prompt, task="final_response_generation", max_new_tokens=512)
#                 except Exception as e:
#                     logging.error(f"Error generating final response: {e}")
#                     final_response_text = "Error while generating the final response."
#         else:
#             # Fallback for no sub-queries
#             final_response_json = {
#                 "user_query": user_query,
#                 "final_response": "No sub-queries found.",
#                 "extra_response": ""
#             }
#             final_response_text = json.dumps(final_response_json, ensure_ascii=False)

#         # Deduplicate documents
#         seen = set()
#         document_contents = []
#         for doc in all_documents:
#             page_content = doc["page_content"] if isinstance(doc, dict) else doc.page_content
#             if page_content not in seen:
#                 seen.add(page_content)
#                 document_contents.append(page_content)

#         # Save final response generation details in the state
#         state.setdefault("keys", {})["final_response_generation"] = {
#             "user_query": user_query,
#             "final_response": final_response_text,
#             "documents": document_contents
#         }

#         # Update chat history with the new message pair
#         await self.chat_history_manager.add_message_pair(session_id, user_query, final_response_text)

#         return state

import json
import logging
from pathlib import Path
from model import TextGenerator
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


class FinalResponseGenerator:
    """
    Generates the final, personalized response by integrating sub-query responses,
    chat history, and the original user query.
    """

    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.prompt_path = current_dir.parent.parent / "system_prompts" / "final_response.txt"
        self.chat_history_manager = ChatHistoryManager()
        self.generator = TextGenerator()

    async def run(self, state: dict) -> dict:
        logging.info("Running final response generation node.")
        user_query = state.get("keys", {}).get("question", "").strip()
        session_id = state.get("keys", {}).get("session_id", "")
        if not session_id:
            logging.error("No session ID found.")
            return state

        chat_history_text = await self.chat_history_manager.format_recent_history_as_text(session_id, max_messages=5)

        sub_query_mapping = state.get("keys", {}).get("sub_query_mapping", {})
        identified_sub_queries = sub_query_mapping.get("sub_queries", [])
        sub_query_answers = sub_query_mapping.get("sub_query_answers", [])
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])

        sub_query_context_parts = []
        all_documents = []
        if sub_query_answers:
            for idx, ans in enumerate(sub_query_answers, start=1):
                completed_query = ans.get("completed_query", "").strip()
                response = ans.get("response", "").strip()
                documents = ans.get("documents", [])
                if completed_query or response:
                    sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
                all_documents.extend(documents)
        elif identified_sub_queries:
            for idx, sq in enumerate(identified_sub_queries, start=1):
                completed_query = sq.get("completed_query", "").strip() if isinstance(sq, dict) else str(sq).strip()
                response = sq.get("response", "").strip() if isinstance(sq, dict) else ""
                documents = sq.get("documents", []) if isinstance(sq, dict) else []
                if completed_query or response:
                    sub_query_context_parts.append(f"Sub-Query {idx}: {completed_query}\nResponse {idx}: {response}")
                all_documents.extend(documents)

        non_inscope_classifications = {"child-abuse", "out-of-scope", "general"}
        for sq in classified_sub_queries:
            classification = sq.get("classification", "")
            if classification in non_inscope_classifications:
                completed_query = sq.get("completed_query", "").strip()
                response = sq.get("response", "").strip()
                documents = sq.get("documents", [])
                if completed_query or response:
                    if not any(ans.get("completed_query") == completed_query for ans in sub_query_answers):
                        sub_query_context_parts.append(f"Sub-Query {len(sub_query_context_parts) + 1}: {completed_query}\nResponse {len(sub_query_context_parts) + 1}: {response}")
                all_documents.extend(documents)

        sub_query_context = "\n\n".join(sub_query_context_parts)

        if classified_sub_queries:
            classifications = [sq["classification"] for sq in classified_sub_queries]
            if "exit-chat" in classifications or all(cls == "out-of-scope" for cls in classifications):
                final_response_json = {
                    "user_query": user_query,
                    "final_response": "",
                    "extra_response": "هل تريد مناقشة موضوع آخر؟" if "exit-chat" in classifications else ""
                }
                final_response_text = json.dumps(final_response_json, ensure_ascii=False)
            else:
                try:
                    with open(self.prompt_path, "r", encoding="utf-8") as f:
                        final_prompt = f.read()
                except Exception as e:
                    logging.error(f"Error reading final response prompt: {e}")
                    final_prompt = ""

                prompt = final_prompt.format(
                    chat_history=chat_history_text,
                    sub_query_context=sub_query_context,
                    user_query=user_query
                )

                logging.info(f"final_response_prompt={prompt}")

                try:
                    final_response_text = self.generator.generate(prompt=prompt, task="final_response_generation", max_new_tokens=512)
                except Exception as e:
                    logging.error(f"Error generating final response: {e}")
                    final_response_text = "Error while generating the final response."

                if contains_chinese(final_response_text):
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
                            {final_response_text}

                            Return only valid JSON in the following format: {{\"translation\": \"<Arabic translation>\"}}
                            """
                        logging.info(f"translation_prompt={translation_prompt}")
                        translated = self.generator.generate(translation_prompt, task="translation")
                        logging.info(f"translated={translated}")
                        if isinstance(translated, list):
                            translated = translated[0]
                        final_response_text = extract_translation_json(translated)
                        logging.info(f"response_text_translated={final_response_text}")
                    except Exception as e:
                        logging.error(f"Error during Chinese-to-Arabic translation of final response: {e}")
                        final_response_text = "تعذر ترجمة المحتوى من الصينية إلى العربية."

        else:
            final_response_json = {
                "user_query": user_query,
                "final_response": "No sub-queries found.",
                "extra_response": ""
            }
            final_response_text = json.dumps(final_response_json, ensure_ascii=False)

        seen = set()
        document_contents = []
        for doc in all_documents:
            page_content = doc["page_content"] if isinstance(doc, dict) else doc.page_content
            if page_content not in seen:
                seen.add(page_content)
                document_contents.append(page_content)

        state.setdefault("keys", {})["final_response_generation"] = {
            "user_query": user_query,
            "final_response": final_response_text,
            "documents": document_contents
        }

        await self.chat_history_manager.add_message_pair(session_id, user_query, final_response_text)

        return state