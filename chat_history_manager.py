import json
import os
import logging
import uuid
import time
import numpy as np
from model import TextGenerator
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

class ChatHistoryManager:
    """
    Manages chat history for each session, supporting both Firebase and local file storage.
    Allows retrieval and updating of chat history using session IDs.
    """
    
    def __init__(self, history_file: str = "chat_history.json"):
        self.history_file = history_file
        self.generator = TextGenerator()
        self.db = self._initialize_firebase()
        
    def _initialize_firebase(self):
        """Initialize Firebase if credentials are available."""
        try:
            # For local testing with service account key file
            if os.path.exists('key.json'):
                cred = credentials.Certificate('key.json')
            # For Vertex AI deployment
            else:
                # If using environment variable
                if os.environ.get('FIREBASE_CREDENTIALS'):
                    cred_dict = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
                    cred = credentials.Certificate(cred_dict)
                # If using mounted secret
                else:
                    cred_path = os.environ.get('FIREBASE_CREDENTIALS_PATH', 'key.json')
                    if os.path.exists(cred_path):
                        cred = credentials.Certificate(cred_path)
                    else:
                        logging.warning(f"Firebase credentials not found at {cred_path}. Using local storage.")
                        return None
            
            # Initialize the app if it's not already initialized
            try:
                app = firebase_admin.get_app()
            except ValueError:
                app = firebase_admin.initialize_app(cred)
                
            db = firestore.client()
            logging.info("Firebase initialized successfully for ChatHistoryManager")
            return db
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {str(e)}")
            return None
    
    def get_session_id(self):
        """Generates a new session ID."""
        return str(uuid.uuid4())

    # Helper functions for embeddings conversion
    def _convert_embeddings_for_storage(self, embeddings):
        """Convert numpy embeddings to a format suitable for Firebase storage."""
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings

    def _convert_embeddings_from_storage(self, embeddings):
        """Convert stored embeddings back to numpy arrays."""
        if isinstance(embeddings, list):
            return np.array(embeddings)
        return embeddings

    async def load_session_data(self, session_id: str) -> dict:
        """Load session data from Firebase or fallback to local file."""
        logging.info(f"Loading session data for session_id: {session_id}")
        default_session_data = {
            "history": [],
            "state": None,
            "service_application_data": {},
        }
        
        # Try to load from Firebase first
        if self.db:
            try:
                doc_ref = self.db.collection('sessions').document(session_id)
                doc = doc_ref.get()
                if doc.exists:
                    session_data = doc.to_dict()
                    logging.info(f"Loaded session data from Firebase for session_id: {session_id}")
                    
                    # Convert stored embeddings back to numpy arrays
                    if "history" in session_data:
                        for item in session_data["history"]:
                            if "embedding" in item:
                                item["embedding"] = self._convert_embeddings_from_storage(item["embedding"])
                    
                    return session_data
                else:
                    logging.info(f"No session data found in Firebase for session_id: {session_id}, creating new session")
                    await self.save_session_data(session_id, default_session_data)
                    return default_session_data
            except Exception as e:
                logging.error(f"Error loading session data from Firebase: {str(e)}")
                # Fallback to file-based storage if Firebase fails
        
        # Fallback to file-based storage
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session_data = data.get(session_id, default_session_data)
                    logging.info(f"Loaded session data from file for session_id: {session_id}")
                    return session_data
            logging.info("No session history file found, returning default session data.")
            return default_session_data
        except Exception as e:
            logging.error(f"Error loading session data from file: {str(e)}")
            return default_session_data

    async def save_session_data(self, session_id: str, session_data: dict):
        """Save session data to Firebase or fallback to local file."""
        logging.info(f"Saving session data for session_id: {session_id}")
        
        # Create a copy to avoid modifying the original
        session_data_copy = session_data.copy()
        
        # Try to save to Firebase first
        if self.db:
            try:
                # Convert numpy arrays to lists for Firebase compatibility
                if "history" in session_data_copy:
                    for item in session_data_copy["history"]:
                        if "embedding" in item:
                            item["embedding"] = self._convert_embeddings_for_storage(item["embedding"])
                
                # Save to Firestore
                doc_ref = self.db.collection('sessions').document(session_id)
                doc_ref.set(session_data_copy)
                logging.info(f"Successfully saved session data to Firebase for session_id: {session_id}")
                return
            except Exception as e:
                logging.error(f"Error saving session data to Firebase: {str(e)}")
                # Fallback to file-based storage if Firebase fails
        
        # Fallback to file-based storage
        try:
            data = {}
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
            # Convert numpy arrays to lists for JSON serialization
            if "history" in session_data_copy:
                for item in session_data_copy["history"]:
                    if "embedding" in item and hasattr(item["embedding"], "tolist"):
                        item["embedding"] = item["embedding"].tolist()
                        
            data[session_id] = session_data_copy
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved session data to file for session_id: {session_id}")
        except Exception as e:
            logging.error(f"Error saving session data to file: {str(e)}")
            raise

    async def get_chat_history(self, session_id: str):
        """
        Retrieve the chat history in a format suitable for the LLM.
        
        Returns:
            dict: Contains 'recent_messages' with user/assistant messages
        """
        session_data = await self.load_session_data(session_id)
        
        # Format the data for LLM use
        recent_messages = []
        if "history" in session_data:
            # Get the most recent conversations (last 5)
            history = session_data["history"]
            recent_history = history[-5:] if len(history) > 5 else history
            
            for item in recent_history:
                if "query" in item and "response" in item:
                    # Add user message
                    recent_messages.append({"role": "user", "content": item["query"]})
                    # Add assistant response
                    recent_messages.append({"role": "assistant", "content": item["response"]})
        
        return {"recent_messages": recent_messages}

    async def add_message_pair(self, session_id: str, user_message: str, assistant_response: str, embedding=None):
        """
        Add a user message and assistant response to the chat history.
        
        Args:
            session_id: The session identifier
            user_message: The user's query
            assistant_response: The assistant's response
            embedding: Optional embedding vector for the user's query
        """
        session_data = await self.load_session_data(session_id)
        
        # Create the history entry
        history_entry = {
            "query": user_message,
            "response": assistant_response,
            "timestamp": time.time()
        }
        
        # Add embedding if provided
        if embedding is not None:
            history_entry["embedding"] = embedding
            
        # Add to history
        if "history" not in session_data:
            session_data["history"] = []
            
        session_data["history"].append(history_entry)
        
        # Save the updated session data
        await self.save_session_data(session_id, session_data)
        
        return session_data
    
    async def format_recent_history_as_text(self, session_id: str, max_messages=5):
        """
        Format recent chat history as text for prompts.
        
        Args:
            session_id: The session identifier
            max_messages: Maximum number of message pairs to include
            
        Returns:
            str: Formatted chat history text
        """
        chat_history = await self.get_chat_history(session_id)
        recent_messages = chat_history.get("recent_messages", [])
        
        history_text = ""
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"]
            history_text += f"{role.upper()}: {content}\n"
            
        return history_text