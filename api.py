import time
import re
import json
import logging
import uuid
import os
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.templating import Jinja2Templates
import uvicorn
import aiofiles  # For async file operations
import asyncio
from model import TextGenerator
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
from config import embeddings  # Assuming this is your LLM configuration
import aiohttp  # For async HTTP requests
from langgraph.types import Command  # Import Command for resumption
import concurrent.futures  # For running synchronous LLM calls in a thread pool
from orchestration.graph_assembly import app as rag_app
from chat_history_manager import ChatHistoryManager

# Firebase imports for session management
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# # Start Phoenix app
session = px.launch_app()

PROJECT_NAME = "arabic-test-6"

tracer_provider = register(
     project_name = PROJECT_NAME,
     endpoint="http://localhost:6006/v1/traces")

LangChainInstrumentor().instrument(tracer_provider = tracer_provider)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize thread pool executor for synchronous LLM calls
executor = concurrent.futures.ThreadPoolExecutor()

# Initialize Firebase
def initialize_firebase():
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
                    logger.warning(f"Firebase credentials not found at {cred_path}. Some functionality will be limited.")
                    return None
                
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        # Return None instead of raising to allow app to start without Firebase in dev
        return None

# Initialize Firebase client
db = initialize_firebase()

class ChatRequest(BaseModel):
    question: str
    session_id: str = None

# Helper functions for embeddings conversion
def convert_embeddings_for_storage(embeddings):
    """Convert numpy embeddings to a format suitable for Firebase storage."""
    if hasattr(embeddings, 'tolist'):
        return embeddings.tolist()
    return embeddings

def convert_embeddings_from_storage(embeddings):
    """Convert stored embeddings back to numpy arrays."""
    if isinstance(embeddings, list):
        return np.array(embeddings)
    return embeddings

async def load_session_data(session_id: str) -> dict:
    """Load session data from Firebase or fallback to local file."""
    logger.info(f"Loading session data for session_id: {session_id}")
    default_session_data = {
        "history": [],
        "state": None,
        "service_application_data": {},
    }
    
    # Try to load from Firebase first
    if db:
        try:
            doc_ref = db.collection('sessions').document(session_id)
            doc = doc_ref.get()
            if doc.exists:
                session_data = doc.to_dict()
                logger.info(f"Loaded session data from Firebase for session_id: {session_id}")
                
                # Convert stored embeddings back to numpy arrays
                if "history" in session_data:
                    for item in session_data["history"]:
                        if "embedding" in item:
                            item["embedding"] = convert_embeddings_from_storage(item["embedding"])
                
                return session_data
            else:
                logger.info(f"No session data found in Firebase for session_id: {session_id}, creating new session")
                await save_session_data(session_id, default_session_data)
                return default_session_data
        except Exception as e:
            logger.error(f"Error loading session data from Firebase: {str(e)}")
            # Fallback to file-based storage if Firebase fails
    
    # Fallback to file-based storage
    try:
        async with aiofiles.open("session_history.json", "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
            session_data = data.get(session_id, default_session_data)
            # Handle old format where session_data is a list
            if isinstance(session_data, list):
                logger.info(
                    f"Converting old list-based session data for session_id: {session_id}"
                )
                session_data = {
                    "history": session_data,  # Treat list as history
                    "state": None,
                    "service_application_data": {},
                }
                # Update the file to new format
                data[session_id] = session_data
                async with aiofiles.open("session_history.json", "w", encoding="utf-8") as f:
                    await f.write(json.dumps(data, ensure_ascii=False))
            logger.info(f"Loaded session data from file for session_id: {session_id}")
            return session_data
    except FileNotFoundError:
        logger.info("No session history file found, returning default session data.")
        return default_session_data
    except Exception as e:
        logger.error(f"Error loading session data from file: {str(e)}")
        return default_session_data

async def save_session_data(session_id: str, session_data: dict):
    """Save session data to Firebase or fallback to local file."""
    logger.info(f"Saving session data for session_id: {session_id}")
    
    # Create a copy to avoid modifying the original
    session_data_copy = session_data.copy()
    
    # Try to save to Firebase first
    if db:
        try:
            # Convert numpy arrays to lists for Firebase compatibility
            if "history" in session_data_copy:
                for item in session_data_copy["history"]:
                    if "embedding" in item:
                        item["embedding"] = convert_embeddings_for_storage(item["embedding"])
            
            # Save to Firestore
            doc_ref = db.collection('sessions').document(session_id)
            doc_ref.set(session_data_copy)
            logger.info(f"Successfully saved session data to Firebase for session_id: {session_id}")
            return
        except Exception as e:
            logger.error(f"Error saving session data to Firebase: {str(e)}")
            # Fallback to file-based storage if Firebase fails
    
    # Fallback to file-based storage
    try:
        try:
            async with aiofiles.open("session_history.json", "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
        except FileNotFoundError:
            logger.info("Creating new session history file.")
            data = {}
        except Exception as e:
            logger.error(f"Error reading session history file: {str(e)}")
            
        # Convert numpy arrays to lists for JSON serialization
        if "history" in session_data_copy:
            for item in session_data_copy["history"]:
                if "embedding" in item and hasattr(item["embedding"], "tolist"):
                    item["embedding"] = item["embedding"].tolist()
                    
        data[session_id] = session_data_copy
        async with aiofiles.open("session_history.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False))
        logger.info(f"Successfully saved session data to file for session_id: {session_id}")
    except Exception as e:
        logger.error(f"Error saving session data to file: {str(e)}")

# Extract conversation history in a format suitable for the RAG pipeline
def extract_conversation_history(session_data, max_history=5):
    """
    Extract the conversation history from session data in a format suitable for the RAG pipeline.
    
    Args:
        session_data (dict): The session data containing history
        max_history (int): Maximum number of conversation turns to include
        
    Returns:
        list: Formatted conversation history
    """
    chat_history = []
    
    if "history" in session_data:
        # Get the most recent conversations
        history = session_data["history"]
        recent_history = history[-max_history:] if len(history) > max_history else history
        
        for item in recent_history:
            # Only include items that have both query and response
            if "query" in item and "response" in item:
                chat_history.append({
                    "user": item["query"],
                    "assistant": item["response"]
                })
    
    return chat_history

# Define the prompt template (unchanged)
response_template = PromptTemplate(
    template=(
        "Provide a response to the user about their service applications based on the following information:\n\n"
        "Service Requests Schema:\n"
        "{schema_str}\n\n"
        "User Query: {query}\n\n"
        "Instructions:\n"
        "- Response should be in arabic."
        "- Read the schema to understand the user's service requests tied to their civil ID.\n"
        "- Use the schema to answer the user's query accurately.\n"
        "- Return a clear and concise response that aligns with the user's query.\n"
        "- If the query cannot be answered with the schema, say so politely.\n\n"
        "- Do not add the User Query: {query} with the response "
        "Response:"
    ),
    input_variables=["schema_str", "query"],
)

def schema_to_str(schema: dict) -> str:
    """Convert the schema dictionary to a readable string format."""
    civil_id = schema["civil_id"]
    requests = schema["service_requests"]
    requests_str = "\n".join(
        [
            f"- ID: {req['id']}, Type: {req['type']}, Status: {req['status']}"
            for req in requests
        ]
    )
    return f"Civil ID: {civil_id}\nService Requests:\n{requests_str}"

async def fetch_service_schema(civil_id: str) -> dict:
    """Simulate an async API call to fetch service schema for a civil_id."""
    logger.info(f"Fetching service schema for civil_id: {civil_id}")
    try:
        # Simulate async API call with aiohttp
        async with aiohttp.ClientSession() as session:
            # For demo, return hardcoded schema (replace with actual API call)
            schema = {
                "civil_id": civil_id,
                "service_requests": [
                    {"id": 1, "type": "renew service", "status": "pending"},
                    {"id": 2, "type": "payment", "status": "completed"},
                    {"id": 3, "type": "required documents", "status": "in progress"},
                ],
            }
            logger.info(f"Retrieved schema: {schema}")
            return schema
    except Exception as e:
        logger.error(f"Error fetching schema: {str(e)}")
        # Fallback schema if API call fails
        return {"civil_id": civil_id, "service_requests": []}

async def get_llm_response(schema: dict, query: str) -> str:
    """Generate an LLM response using a prompt template, schema, and user query asynchronously."""
    logger.info(f"Generating LLM response for query: {query}")

    # Convert schema to string
    try:
        schema_str = schema_to_str(schema)
        logger.info(f"Formatted schema: {schema_str}")
    except Exception as e:
        logger.error(f"Error formatting schema: {str(e)}")
        return "Sorry, I couldn't process your request due to an internal error."

    # Format the prompt with schema and query
    try:
        prompt = response_template.format(schema_str=schema_str, query=query)
        logger.info(f"Formatted prompt: {prompt}")
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        return "Sorry, I couldn't process your request due to an internal error."

    # Call the LLM asynchronously (assuming llama_llm.invoke is synchronous)
    try:
        loop = asyncio.get_event_loop()
        generator = TextGenerator()
        response = await loop.run_in_executor(
            executor,
            lambda: generator.generate(prompt, task="service", max_new_tokens=128)
        )
        logger.info(f"LLM response: {response}")
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}")
        response = f"I'm sorry, but I couldn't find a service request matching your query: {query}."

    return response


import json
import logging

def extract_final_response(raw: str) -> str:
    """
    Extracts the 'final_response' value from a JSON object hidden inside any wrapper string.
    Handles arbitrarily nested braces without external regex libraries.
    Falls back to returning the raw input if extraction or parsing fails.
    """
    if not raw:
        return ""

    # 1) Locate the first '{'
    start = raw.find("{")
    if start == -1:
        logging.warning("No JSON object found; returning raw string.")
        return raw.strip()

    # 2) Find the matching '}' by counting nesting depth
    depth = 0
    end = None
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    # 3) Slice out the JSON substring
    json_str = raw[start:end] if end is not None else raw[start:]

    # 4) Parse JSON
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}; returning raw string.")
        return raw.strip()

    # 5) Return the final_response field (or fall back)
    final = payload.get("final_response")
    if final is None:
        logging.warning("'final_response' key missing; returning raw string.")
        return raw.strip()

    return final

async def run_advanced_rag_pipeline(question: str, session_id: str) -> dict:
    logger.info(f"Processing query: {question} for session_id: {session_id}")

    # Load session data
    history_manager = ChatHistoryManager()
    # session_data = await load_session_data(session_id)
    session_data = await history_manager.load_session_data(session_id)
    state = session_data.get("state", None)

    # Handle service-application states
    if state == "awaiting_civil_id":
        civil_id = question  # Treat query as civil ID
        session_data["service_application_data"]["civil_id"] = civil_id
        session_data["state"] = "awaiting_phone_number"
        await history_manager.save_session_data(session_id, session_data)
        return {
            "primary_response": "يرجى تقديم رقم هاتفك.",
            "primary_classification": "service-application",
            "secondary_response": None,
            "secondary_classification": None,
            "show_buttons": False,
            "button_options": [],
        }
    elif state == "awaiting_phone_number":
        phone_number = question  # Treat query as phone number
        session_data["service_application_data"]["phone_number"] = phone_number
        session_data["state"] = "awaiting_otp"
        await history_manager.save_session_data(session_id, session_data)
        return {
            "primary_response": "يرجى تقديم رمز OTP.",
            "primary_classification": "service-application",
            "secondary_response": None,
            "secondary_classification": None,
            "show_buttons": False,
            "button_options": [],
        }
    elif state == "awaiting_otp":
        otp = question  # Treat query as OTP
        session_data["service_application_data"]["otp"] = otp
        civil_id = session_data["service_application_data"]["civil_id"]
        original_query = session_data["service_application_data"]["original_query"]
        # Fetch schema
        schema = await fetch_service_schema(civil_id)
        final_answer = await get_llm_response(schema, original_query)
        session_data["state"] = None  # Reset state
        # Add original query to history
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            executor, lambda: embeddings.embed_query(original_query)
        )
        session_data["history"].append(
            {
                "query": original_query, 
                "embedding": query_embedding,
                "response": final_answer,
                "timestamp": time.time()
            }
        )
        await history_manager.save_session_data(session_id, session_data)
        return {
            "primary_response": final_answer,
            "primary_classification": "service-application",
            "secondary_response": None,
            "secondary_classification": None,
            "show_buttons": False,
            "button_options": [],
        }

    # Extract conversation history to provide context to the RAG pipeline
    chat_history = extract_conversation_history(session_data)
    logger.info(f"Extracted {len(chat_history)} conversation turns for context")

    # Generate embedding for query
    logger.info("Generating embedding for query...")
    try:
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            executor, lambda: embeddings.embed_query(question)
        )
        logger.info(f"Embedding generated successfully, length: {len(query_embedding)}")
    except Exception as e:
        logger.error(f"Failed to generate embedding for query: {str(e)}")
        return {
            "primary_response": "Internal server error: Failed to generate embedding.",
            "primary_classification": "out-of-scope",
            "secondary_response": None,
            "secondary_classification": None,
            "show_buttons": False,
            "button_options": [],
        }

    # Load session history for similarity check
    session_history = session_data.get("history", [])
    previous_embeddings = [q["embedding"] for q in session_history]
    if previous_embeddings:
        logger.info(
            f"Found {len(previous_embeddings)} previous queries for similarity check."
        )
        previous_embeddings = np.array(previous_embeddings)
        similarities = cosine_similarity([query_embedding], previous_embeddings)[0]
        similar_count = np.sum(similarities >= 0.9)
        logger.info(
            f"Similarity check completed, {similar_count} queries with similarity >= 0.9"
        )
        for idx, (sim_score, prev_query) in enumerate(
            zip(similarities, [q["query"] for q in session_history])
        ):
            logger.info(
                f"Query: {question} vs Previous Query: {prev_query}, Similarity Score: {sim_score:.4f}"
            )
        if similar_count >= 2:
            response = "لقد لاحظت أنك سألت سؤالًا عدة مرات. هل ترغب في متابعة الدردشة أو الاتصال بوكيل بشري؟"
            classification = "human-agent"
            show_buttons = True
            button_options = [
                "Yes, Please Connect the call with a human-agent",
                "No, continue the chat",
            ]
            logger.info("Escalating to human-agent due to repeated similar queries.")
            session_data["history"].append(
                {
                    "query": question, 
                    "embedding": query_embedding,
                    "response": response,
                    "timestamp": time.time()
                }
            )
            await history_manager.save_session_data(session_id, session_data)
            return {
                "primary_response": response,
                "primary_classification": classification,
                "secondary_response": None,
                "secondary_classification": None,
                "show_buttons": show_buttons,
                "button_options": button_options,
            }
    else:
        logger.info("No previous queries found for similarity check.")

    # Proceed with RAG pipeline using streaming with interrupt support
    logger.info("Running RAG pipeline with interrupt support...")
    inputs = {
        "keys": {
            "question": question, 
            "session_id": session_id,
            "chat_history": chat_history  # Pass conversation history to the pipeline
        }
    }
    thread = {"configurable": {"thread_id": session_id}}
    final_state = None

    async for event in rag_app.astream(inputs, thread, stream_mode="updates"):
        if "_interrupt_" in event:
            interrupt_data = event["_interrupt_"]
            clarifying_question = (
                interrupt_data[0].value
                if isinstance(interrupt_data, (list, tuple))
                else str(interrupt_data)
            )
            logger.info(f"Interrupt detected, clarifying question: {clarifying_question}")
            session_data["state"] = "rag_interrupted"
            session_data["clarifying_question"] = clarifying_question
            await history_manager.save_session_data(session_id, session_data)
            return {
                "primary_response": clarifying_question,
                "primary_classification": "clarification_needed",
                "secondary_response": None,
                "secondary_classification": None,
                "show_buttons": False,
                "button_options": [],
            }
        else:
            for step_name, step_data in event.items():
                if not isinstance(step_data, dict):
                    continue
                final_state = step_data.get("keys", {})
    
   
    #if final_state and "response" in final_state:
    # only fire this if exactly one in-scope sub-query was present
    if final_state and final_state.get("only_one_in_scope", False) and "response" in final_state:
        raw = final_state["response"]
        final_text = extract_final_response(raw)  # e.g. "<|response|> { ... }"
        logging.info(f"raw={raw}")
        logging.info(f"final_text={final_text}")

        # ─── NEW: append to history and save session_data ───
        session_data["history"].append({
            "query": question,             # the user’s input
            "embedding": query_embedding,   # the embedding you generated earlier
            "response": final_text,         # the extracted final text
            "timestamp": time.time()
        })
        await history_manager.save_session_data(session_id, session_data)

        return {
            "primary_response":        final_text,
            "primary_classification":  "in-scope",
            "secondary_response":      None,
            "secondary_classification": None,
            "show_buttons":            False,
            "button_options":          [],
        }
    if final_state and final_state.get("classification") == "general" and "response" in final_state:
        raw = final_state["response"]  # this is your JSON string
        logging.info(f"raw={raw}")
        try:
            payload = json.loads(raw)
            final_text = payload.get("final_response", raw)
            logging.info(f"final_test: {final_text}")
        except json.JSONDecodeError:
            final_text = raw  # fallback if it's not valid JSON

        # ─── NEW: append “general” turn to history ───
        session_data["history"].append({
            "query":     question,
            "embedding": query_embedding,
            "response":  final_text,
            "timestamp": time.time()
        })
        await history_manager.save_session_data(session_id, session_data)

        return {
            "primary_response":       final_text,
            "primary_classification": "general",
            "secondary_response":     None,
            "secondary_classification": None,
            "show_buttons":           False,
            "button_options":         [],
        }
    if final_state and "escalation_message" in final_state:
        classification = final_state.get("classification", "out-of-scope")
        msg = final_state["escalation_message"]
        buttons = final_state.get("button_options", [])
        show_buttons = final_state.get("show_buttons", False)

        # *NEW*: append “escalation” turn to history
        session_data["history"].append({
            "query": question, 
            "embedding": query_embedding,    # keep the same embedding you generated earlier
            "response": msg,                 # store the escalation message as “response”
            "timestamp": time.time()
        })
        await history_manager.save_session_data(session_id, session_data)

        # *NEW*: if we're in service-application, record the next step
        if classification == "service-application":
            session_data["state"] = "awaiting_civil_id"
            # stash the original query so we can use it after OTP
            session_data["service_application_data"] = {"original_query": question}
            await history_manager.save_session_data(session_id, session_data)

        return {
            "primary_response":       msg,
            "primary_classification": classification,
            "secondary_response":     None,
            "secondary_classification":None,
            "show_buttons":           show_buttons,
            "button_options":         buttons,
        }

    if final_state is None:
        logger.warning("RAG pipeline produced no final state.")
        response = "No final generation produced."
        classification = "out-of-scope"
        show_buttons = False
        button_options = []
    else:
        final_answer = final_state.get(
            "final_response_generation", "No final generation produced."
        )
        logger.info(f"Raw final_answer: {final_answer}")

        sub_query_mapping = final_state.get("sub_query_mapping", {})
        classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
        all_classifications = []
        if classified_sub_queries:
            classifications = [sq["classification"] for sq in classified_sub_queries]
            logger.info(f"Sub-query classifications: {classifications}")
            if "in-scope" in classifications:
                all_classifications.append("in-scope")
            if "exit-chat" in classifications:
                all_classifications.append("exit-chat")
            if "human-agent" in classifications:
                all_classifications.append("human-agent")
            if "out-of-scope" in classifications:
                all_classifications.append("out-of-scope")
            if "service-application" in classifications:
                all_classifications.append("service-application")
            if 'child-abuse' in classifications:
                all_classifications.append("child-abuse")
            if not all_classifications:
                all_classifications.append("general")
        else:
            default_classification = final_state.get("classification", "out-of-scope")
            all_classifications.append(default_classification)
        logger.info(f"All classifications detected: {all_classifications}")

        # Initialize response variables
        primary_response = ""
        primary_classification = ""
        secondary_response = None
        secondary_classification = None
        show_buttons = False
        button_options = []

        logging.info(f"final_state={final_state}")
        multiple_no_inscope = final_state.get("multiple_no_inscope", False)
        logging.info(f"multiple_no_inscope={multiple_no_inscope}")
        # Handle classifications
        if "in-scope" in all_classifications:
            primary_response = final_answer
            primary_classification = "in-scope"
            if "service-application" in all_classifications:
                secondary_classification = "service-application"
                secondary_response = "الآن، للمتابعة في طلب الخدمة، يرجى تقديم رقم هويتك المدنية."
                session_data["state"] = "awaiting_civil_id"
                session_data["service_application_data"] = {"original_query": question}
                await history_manager.save_session_data(session_id, session_data)
            else:
                if "exit-chat" in all_classifications:
                    secondary_classification = "exit-chat"
                    secondary_response = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
                    show_buttons = True
                    button_options = ["Do you want to start over?", "Continue"]
                elif "human-agent" in all_classifications:
                    secondary_classification = "human-agent"
                    secondary_response = "هل تريد الاتصال بوكيل بشري؟"
                    show_buttons = True
                    button_options = [
                        "Yes, Please Connect the call with a human-agent",
                        "No, continue the chat",
                    ]
                    
        elif multiple_no_inscope:
            logger.info("Detected multiple_no_inscope=True, handling multi-non-inscope case")
            primary_response = final_answer
            primary_classification = 'multiple-no-inscope'
            if "service-application" in all_classifications:
                secondary_classification = "service-application"
                secondary_response = "الآن، للمتابعة في طلب الخدمة، يرجى تقديم رقم هويتك المدنية."
                session_data["state"] = "awaiting_civil_id"
                session_data["service_application_data"] = {"original_query": question}
                await history_manager.save_session_data(session_id, session_data)
            else:
                if "exit-chat" in all_classifications:
                    secondary_classification = "exit-chat"
                    secondary_response = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
                    show_buttons = True
                    button_options = ["Do you want to start over?", "Continue"]
                elif "human-agent" in all_classifications:
                    secondary_classification = "human-agent"
                    secondary_response = "هل تريد الاتصال بوكيل بشري؟"
                    show_buttons = True
                    button_options = [
                        "Yes, Please Connect the call with a human-agent",
                        "No, continue the chat",
                    ]

        elif "child-abuse" in all_classifications:
            primary_response = "يرجى العثور على رابط نموذج إساءة معاملة الأطفال: https://portal.mosd.gov.om/webcenter/portal/MOSDExternalPortal/pages_services/reportabuse"
            primary_classification = "child-abuse"
            show_buttons = False
            button_options = []
        elif "service-application" in all_classifications:
            primary_response = "يرجى تقديم رقم الهوية المدنية."
            primary_classification = "service-application"
            session_data["state"] = "awaiting_civil_id"
            session_data["service_application_data"] = {"original_query": question}
            await history_manager.save_session_data(session_id, session_data)
        elif "exit-chat" in all_classifications:
            primary_response = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
            primary_classification = "exit-chat"
            show_buttons = True
            button_options = ["Do you want to start over?", "Continue"]
        elif "human-agent" in all_classifications:
            primary_response = "هل تريد الاتصال بوكيل بشري؟"
            primary_classification = "human-agent"
            show_buttons = True
            button_options = [
                "Yes, Please Connect the call with a human-agent",
                "No, continue the chat",
            ]
        elif "out-of-scope" in all_classifications:
            primary_classification = "out-of-scope"
            primary_response = "نعتذر، الموضوع خارج اختصاص عمل الوزارة"
        else:
            primary_classification = "general"
            primary_response = final_answer

        # Prepare the final answer and process it for saving
        # final_answer = primary_response["final_response"]
        if isinstance(primary_response, str):
            final_answer = primary_response
        else:
            final_answer = primary_response["final_response"]
        if secondary_response:
            logging.info(f"final_answer={final_answer}, secondary_response={secondary_response}")
            final_answer += "\n\n" + secondary_response

        # Save query to history if not in service-application flow
        if session_data.get("state") is None:
            session_data["history"].append(
                {
                    "query": question, 
                    "embedding": query_embedding,
                    "response": primary_response,  # Store the response for conversation history
                    "timestamp": time.time()
                }
            )
            await history_manager.save_session_data(session_id, session_data)

        response_info = {
            "primary_response": primary_response,
            "primary_classification": primary_classification,
            "secondary_response": secondary_response,
            "secondary_classification": secondary_classification,
            "show_buttons": show_buttons,
            "button_options": button_options,
        }
        logger.info(f"Response info prepared: {response_info}")
        return response_info

    # Default return
    session_data["history"].append({
        "query": question, 
        "embedding": query_embedding,
        "response": "No final generation produced.",
        "timestamp": time.time()
    })
    await history_manager.save_session_data(session_id, session_data)
    return {
        "primary_response": "No final generation produced.",
        "primary_classification": "out-of-scope",
        "secondary_response": None,
        "secondary_classification": None,
        "show_buttons": False,
        "button_options": [],
    }

async def process_query(query: str, session_id: str) -> dict:
    logger.info(
        f"Starting query processing for query: {query}, session_id: {session_id}"
    )
    start = time.perf_counter()
    session_data = await load_session_data(session_id)
    state = session_data.get("state", None)

    if state == "rag_interrupted":
        # Extract conversation history for context
        chat_history = extract_conversation_history(session_data)
        logger.info(f"Extracted {len(chat_history)} conversation turns for resumption context")
        
        # Resume pipeline with clarification
        clarification = query
        thread = {"configurable": {"thread_id": session_id}}
        cmd = Command(resume=clarification)
        final_state = None
        async for event in rag_app.astream(cmd, thread, stream_mode="updates"):
            for step_name, step_data in event.items():
                if not isinstance(step_data, dict):
                    continue
                final_state = step_data.get("keys", {})
        
        if final_state is None:
            logger.warning("RAG pipeline produced no final state after resumption.")
            response_info = {
                "primary_response": "No final generation produced after clarification.",
                "primary_classification": "out-of-scope",
                "secondary_response": None,
                "secondary_classification": None,
                "show_buttons": False,
                "button_options": [],
            }
        else:
            final_answer = final_state.get(
                "final_response_generation", "No final generation produced."
            )
            logger.info(f"Raw final_answer after resumption: {final_answer}")

            sub_query_mapping = final_state.get("sub_query_mapping", {})
            classified_sub_queries = sub_query_mapping.get("classified_sub_queries", [])
            all_classifications = []
            if classified_sub_queries:
                classifications = [sq["classification"] for sq in classified_sub_queries]
                logger.info(f"Sub-query classifications: {classifications}")
                if "in-scope" in classifications:
                    all_classifications.append("in-scope")
                if 'child-abuse' in classifications:
                    all_classifications.append("child-abuse")
                if "exit-chat" in classifications:
                    all_classifications.append("exit-chat")
                if "human-agent" in classifications:
                    all_classifications.append("human-agent")
                if "out-of-scope" in classifications:
                    all_classifications.append("out-of-scope")
                if "service-application" in classifications:
                    all_classifications.append("service-application")
                if not all_classifications:
                    all_classifications.append("general")
            else:
                default_classification = final_state.get("classification", "out-of-scope")
                all_classifications.append(default_classification)
            logger.info(f"All classifications detected: {all_classifications}")

            # Initialize response variables
            primary_response = ""
            primary_classification = ""
            secondary_response = None
            secondary_classification = None
            show_buttons = False
            button_options = []

            # Handle classifications
            if "in-scope" in all_classifications:
                primary_response = final_answer
                primary_classification = "in-scope"
                if "service-application" in all_classifications:
                    secondary_classification = "service-application"
                    secondary_response = "\n\nNow, to proceed with your service application, please provide your civil ID."
                    session_data["state"] = "awaiting_civil_id"
                    session_data["service_application_data"] = {"original_query": query}
                    await save_session_data(session_id, session_data)
                else:
                    if "exit-chat" in all_classifications:
                        secondary_classification = "exit-chat"
                        secondary_response = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
                        show_buttons = True
                        button_options = ["Do you want to start over?", "Continue"]
                    elif "human-agent" in all_classifications:
                        secondary_classification = "human-agent"
                        secondary_response = "هل تريد الاتصال بوكيل بشري؟"
                        show_buttons = True
                        button_options = [
                            "Yes, Please Connect the call with a human-agent",
                            "No, continue the chat",
                        ]
            elif "service-application" in all_classifications:
                primary_response = "يرجى تقديم رقم الهوية المدنية."
                primary_classification = "service-application"
                session_data["state"] = "awaiting_civil_id"
                session_data["service_application_data"] = {"original_query": query}
                await save_session_data(session_id, session_data)
            elif "child-abuse" in all_classifications:
                primary_response = "يرجى العثور على رابط نموذج إساءة معاملة الأطفال: https://portal.mosd.gov.om/webcenter/portal/MOSDExternalPortal/pages_services/reportabuse"
                primary_classification = "child-abuse"
                show_buttons = False
                button_options = []
            elif "exit-chat" in all_classifications:
                primary_response = "شكرًا على المحادثة! ماذا تود أن تفعل بعد ذلك؟"
                primary_classification = "exit-chat"
                show_buttons = True
                button_options = ["Do you want to start over?", "Continue"]
            elif "human-agent" in all_classifications:
                primary_response = "هل تريد الاتصال بوكيل بشري؟"
                primary_classification = "human-agent"
                show_buttons = True
                button_options = [
                    "Yes, Please Connect the call with a human-agent",
                    "No, continue the chat",
                ]
            elif "out-of-scope" in all_classifications:
                primary_classification = "out-of-scope"
                primary_response = "نعتذر، الموضوع خارج اختصاص عمل الوزارة"
                primary_classification = "general"
                primary_response = final_answer

            response_info = {
                "primary_response": primary_response,
                "primary_classification": primary_classification,
                "secondary_response": secondary_response,
                "secondary_classification": secondary_classification,
                "show_buttons": show_buttons,
                "button_options": button_options,
            }
            
            # Save the clarification and response to history
            # Generate embedding for clarification
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                executor, lambda: embeddings.embed_query(query)
            )
            
            session_data["history"].append(
                {
                    "query": query, 
                    "embedding": query_embedding,
                    "response": primary_response,  # Store the response for conversation history
                    "timestamp": time.time()
                }
            )
        
        # Reset state after resumption
        session_data["state"] = None
        await save_session_data(session_id, session_data)
    else:
        response_info = await run_advanced_rag_pipeline(query, session_id)

    end = time.perf_counter()
    logger.info(f"[Pipeline] Total elapsed time: {end - start:.3f}s")
    logger.info(f"Query processing completed. Response info: {response_info}")
    return response_info

# Setup FastAPI app
app = FastAPI()

# Only mount static files if directory exists (might not be available in Vertex AI)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
    
    @app.get("/", response_class=HTMLResponse)
    async def get_index(request: Request):
        logger.info("Serving chatbot UI at root URL")
        return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint for Vertex AI
@app.get("/health")
def health_check():
    """Health check endpoint for Vertex AI."""
    return {"status": "healthy"}

# Existing chat endpoint for local testing and normal API usage
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    session_id = chat_request.session_id or str(uuid.uuid4())
    query = chat_request.question
    logger.info(f"Received chat request: query={query}, session_id={session_id}")

    response_info = await process_query(query, session_id)

    if isinstance(response_info, dict):
        primary_response = response_info["primary_response"]
        primary_classification = response_info["primary_classification"]
        secondary_response = response_info["secondary_response"]
        secondary_classification = response_info["secondary_classification"]
        show_buttons = response_info["show_buttons"]
        button_options = response_info["button_options"]
    else:
        primary_response, primary_classification, show_buttons, button_options = (
            response_info
        )
        secondary_response = None
        secondary_classification = None

    primary_answer = ""
    if primary_classification in [
        "exit-chat",
        "out-of-scope",
        "human-agent",
        "service-application",
        "clarification_needed",
        "child-abuse",
    ]:
        primary_answer = primary_response
    else:
        if isinstance(primary_response, dict):
            inner_json = primary_response.get(
                "final_response", "لم يتم العثور على رد نهائي."
            )
            if isinstance(inner_json, str):
                try:
                    inner_json = inner_json.encode().decode("utf-8")
                    parsed_inner_json = json.loads(inner_json)
                    logger.info(f"Parsed inner JSON: {parsed_inner_json}")
                    final_response = parsed_inner_json.get(
                        "final_response", "لم يتم العثور على رد نهائي."
                    )
                    extra_response = parsed_inner_json.get("extra_response", "")
                    logger.info(
                        f"Extracted final_response: {final_response}, extra_response: {extra_response}"
                    )
                    primary_answer = str(final_response)
                    if extra_response:
                        primary_answer += "\n\n" + str(extra_response)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Inner JSON parsing failed: {e}, inner_json: {inner_json}"
                    )
                    final_response_match = re.search(
                        r'"final_response":\s*"([^"]+)"', inner_json
                    )
                    extra_response_match = re.search(
                        r'"extra_response":\s*"([^"]+)"', inner_json
                    )
                    final_response = (
                        final_response_match.group(1)
                        if final_response_match
                        else "لم يتم العثور على رد نهائي."
                    )
                    extra_response = (
                        extra_response_match.group(1) if extra_response_match else ""
                    )
                    logger.info(
                        f"Fallback extracted final_response: {final_response}, extra_response: {extra_response}"
                    )
                    primary_answer = str(final_response)
                    if extra_response:
                        primary_answer += "\n\n" + str(extra_response)
            else:
                logger.warning(
                    f"Unexpected type for inner_json: {type(inner_json)}, inner_json: {inner_json}"
                )
                primary_answer = str(inner_json)
        else:
            logger.warning(
                f"Unexpected type for primary_response: {type(primary_response)}, primary_response: {primary_response}"
            )
            primary_answer = str(primary_response)

    final_answer = primary_answer
    if secondary_response:
        final_answer += "\n\n" + secondary_response

    logger.info(f"Final answer: {final_answer}")

    response_content = {
        "answer": final_answer,
        "session_id": session_id,
        "show_buttons": show_buttons,
        "classification": (
            primary_classification
            if not secondary_classification
            else f"{primary_classification},{secondary_classification}"
        ),
        "button_options": button_options,
    }
    logger.info(f"Sending response: {response_content}")
    return JSONResponse(content=response_content)

# New predict endpoint for Vertex AI
@app.post("/predict")
async def predict(request: Request):
    """Endpoint for Vertex AI prediction."""
    try:
        # Parse the request body
        request_json = await request.json()
        
        # Extract the inputs - Vertex AI sends instances
        instances = request_json.get('instances', [])
        if not instances:
            return JSONResponse(
                status_code=400,
                content={"error": "No instances provided in the request"}
            )
        
        # Process each instance
        responses = []
        for instance in instances:
            query = instance.get('query', '')
            session_id = instance.get('session_id', str(uuid.uuid4()))
            
            if not query:
                responses.append({
                    "error": "No query provided in the instance",
                    "session_id": session_id
                })
                continue
            
            # Process the query
            response_info = await process_query(query, session_id)
            
            # Format the response (reusing the logic from chat_endpoint)
            if isinstance(response_info, dict):
                primary_response = response_info["primary_response"]
                primary_classification = response_info["primary_classification"]
                secondary_response = response_info["secondary_response"]
                secondary_classification = response_info["secondary_classification"]
                show_buttons = response_info["show_buttons"]
                button_options = response_info["button_options"]
            else:
                primary_response, primary_classification, show_buttons, button_options = (
                    response_info
                )
                secondary_response = None
                secondary_classification = None
            
            # Process the response content (reusing the logic from chat_endpoint)
            primary_answer = ""
            if primary_classification in [
                "exit-chat",
                "out-of-scope",
                "human-agent",
                "service-application",
                "clarification_needed",
                "child-abuse"
            ]:
                primary_answer = primary_response
            else:
                if isinstance(primary_response, dict):
                    inner_json = primary_response.get(
                        "final_response", "لم يتم العثور على رد نهائي."
                    )
                    if isinstance(inner_json, str):
                        try:
                            inner_json = inner_json.encode().decode("utf-8")
                            parsed_inner_json = json.loads(inner_json)
                            final_response = parsed_inner_json.get(
                                "final_response", "لم يتم العثور على رد نهائي."
                            )
                            extra_response = parsed_inner_json.get("extra_response", "")
                            primary_answer = str(final_response)
                            if extra_response:
                                primary_answer += "\n\n" + str(extra_response)
                        except json.JSONDecodeError as e:
                            final_response_match = re.search(
                                r'"final_response":\s*"([^"]+)"', inner_json
                            )
                            extra_response_match = re.search(
                                r'"extra_response":\s*"([^"]+)"', inner_json
                            )
                            final_response = (
                                final_response_match.group(1)
                                if final_response_match
                                else "لم يتم العثور على رد نهائي."
                            )
                            extra_response = (
                                extra_response_match.group(1) if extra_response_match else ""
                            )
                            primary_answer = str(final_response)
                            if extra_response:
                                primary_answer += "\n\n" + str(extra_response)
                    else:
                        primary_answer = str(inner_json)
                else:
                    primary_answer = str(primary_response)
            
            final_answer = primary_answer
            if secondary_response:
                final_answer += "\n\n" + secondary_response
            
            response_content = {
                "answer": final_answer,
                "session_id": session_id,
                "show_buttons": show_buttons,
                "classification": (
                    primary_classification
                    if not secondary_classification
                    else f"{primary_classification},{secondary_classification}"
                ),
                "button_options": button_options,
            }
            
            responses.append(response_content)
        
        # Return predictions in Vertex AI expected format
        return JSONResponse(content={"predictions": responses})
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

# Firebase connection test endpoint
@app.get("/test-firebase")
async def test_firebase():
    """Test endpoint for Firebase connection."""
    if not db:
        return {"status": "error", "message": "Firebase not initialized"}
    
    try:
        test_id = f"test-{uuid.uuid4()}"
        test_data = {"test": True, "timestamp": time.time()}
        
        # Test write
        doc_ref = db.collection('test').document(test_id)
        doc_ref.set(test_data)
        
        # Test read
        doc = doc_ref.get()
        if doc.exists:
            # Clean up
            doc_ref.delete()
            return {"status": "success", "message": "Firebase connection working"}
        else:
            return {"status": "error", "message": "Failed to read test document"}
    except Exception as e:
        return {"status": "error", "message": f"Firebase test failed: {str(e)}"}

async def main():
    """Main function to run the FastAPI application with cleanup."""
    port = int(os.environ.get("PORT", 8080))  # Vertex AI uses PORT env variable
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    try:
        logger.info(f"Starting FastAPI application on port {port}...")
        await server.serve()
    finally:
        # Cleanup thread pool executor
        executor.shutdown(wait=True)
        logger.info("Thread pool executor shut down.")

if __name__ == "__main__":
    asyncio.run(main())