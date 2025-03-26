import json
import os
import logging
from .config import llama_llm

class ChatHistoryManager:
    """
    Manages chat history for each session, storing it in a local file.
    Allows retrieval and updating of chat history using session IDs.
    """
    
    def __init__(self, history_file: str = "chat_history.json"):
        self.history_file = history_file
        self.chat_history = self.load_chat_history()
    
    
    def get_session_id(self):
        """Generates a new session ID or returns the existing one."""
        return str(uuid.uuid4())  # Generate a new session ID if none exists


    def load_chat_history(self):
        """
        Loads chat history from a file if it exists. If not, returns an empty dictionary.
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as file:
                    return json.load(file)
            except Exception as e:
                logging.error(f"Error loading chat history: {e}")
                return {}
        return {}

    def save_chat_history(self):
        """
        Saves the current chat history to a file.
        """
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.chat_history, file, indent=4)
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")

    def get_chat_history(self, session_id: str):
        """
        Retrieve the chat history for a specific session ID.
        If session doesn't exist, return an empty dictionary.
        """
        return self.chat_history.get(session_id, {"older_summary": "", "recent_messages": []})

    def update_chat_history(self, session_id: str, user_message: str, assistant_message: str):
        """
        Update the chat history with new user and assistant messages.
        """
        session_data = self.get_chat_history(session_id)
        
        recent_messages = session_data.get("recent_messages", [])
        older_summary = session_data.get("older_summary", "")
        max_n = 5  # Number of recent messages to keep without summarizing

        # Append the new messages to recent_messages
        if user_message:
            recent_messages.append({"role": "user", "content": user_message})

        if assistant_message:
            recent_messages.append({"role": "assistant", "content": assistant_message})

        # If the number of recent messages exceeds the max_n, summarize the older messages
        if len(recent_messages) > max_n:
            messages_to_summarize = recent_messages[:-max_n]
            recent_messages = recent_messages[-max_n:]
            summary_of_older = self.summarize(messages_to_summarize, older_summary)
            older_summary = summary_of_older

        # Update session data
        session_data["recent_messages"] = recent_messages
        session_data["older_summary"] = older_summary
        self.chat_history[session_id] = session_data
        
        # Save the updated chat history
        self.save_chat_history()

    def summarize(self, messages_to_summarize, older_summary):
        """
        Summarize older chat history.
        """
        text_to_summarize = older_summary + "\n\n"
        for msg in messages_to_summarize:
            role = msg["role"]
            content = msg["content"]
            text_to_summarize += f"{role.upper()}: {content}\n"

        # You can use your LLM to generate the summary.
        # Example:
        prompt = f"""Summarize the following conversation history into a concise but complete summary:
{text_to_summarize}
---
Return only the summary with no extra text:
"""
        try:
            # Replace with your LLM call to generate the summary
            response = llama_llm.invoke(([("system", prompt)]))
            return response.content.strip()
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return older_summary  # Fallback

