import time

from dotenv import load_dotenv
import os

from pgvector_store import new_session, add_messages, get_session_history, search_session
from retrieval import search_similar_chunks
from rag_prompt import format_chunks_as_context

import requests

load_dotenv()  # Load environment variables from .env file
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API")
GEMINI_API_KEY = os.getenv("GEMINI_API")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


SYSTEM_MESSAGE = """You are a helpful fitness and wellness expert assistant. 
Your role is to answer questions about wellness, fitness, and nutrition based ONLY on the provided knowledge base.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain information to answer the question, say "I don't have enough information to answer this"
3. Be specific and cite which part of the context you're using
4. Keep your answer clear and concise
5. Do not add information from outside sources
6. answer in a formal and friendly tone, suitable for a general audience
7. use the history of the conversation to provide context for your answers, but do not make up information"""


MIN_THRESHOLD = -1
class ConversationSession:
    # def __init__(self):
    #     try:
    #         self.session_id=new_session()
    #     except Exception as e:
    #         print(f"Error creating new session: {e}")
    #         print
    #         self.session_id = None
    #     self.history=[]
    #constructor given a session_id, load the history from the database
    def __init__(self, session_id: str=None):
        if session_id is None:
            try:
                self.session_id = new_session()
            except Exception as e:
                print(f"Error creating new session: {e}")
                self.session_id = None
            self.history = []
        else :
            
            if search_session(session_id) is False:
                print(f"Session ID {session_id} not found in database. Starting a new session.")
                try:
                    self.session_id = new_session()
                except Exception as e:
                    print(f"Error creating new session: {e}")
                    self.session_id = None
                self.history = []
            else:
                self.session_id = session_id
                try:
                    self.history = get_session_history(session_id)
                    print(f"Loaded history for session {session_id}: {self.history}")
                except Exception as e:                
                    print(f"Error loading session history: {e}")
                    self.history = []
    def ask(self, question: str) -> tuple[str, list, float, float]:
        retrieval_start = time.time()
        chunks = search_similar_chunks(question)
        generation_start = time.time()
        retrieval_time = generation_start - retrieval_start
        #skipping LLM call in case of low similarity to save requests, tokens and time.
        if chunks and chunks[0][-1] < MIN_THRESHOLD:
            answer = "I don't have enough information to answer this question based on the provided context."

            if self.session_id is not None:
                try:
                    messages=[{"role": "user", "content": question}, {"role": "model", "content": answer}]
                    add_messages(self.session_id, messages)
                except Exception as e:
                    print(f"Error adding messages: {e}")
            else:
                print("messages not added to the database")
            self.history.append({"role": "user", "content": question})  # store raw
            self.history.append({"role": "model", "content": answer})
            generation_time = time.time() - generation_start
            return answer, chunks, retrieval_time, generation_time
        context = format_chunks_as_context(chunks)
        user_turn = {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        messages_to_send = self.history + [user_turn]
        answer = call_llm(messages_to_send)
        if self.session_id is not None:
            try:
                messages=[{"role": "user", "content": question}, {"role": "model", "content": answer}]
                add_messages(self.session_id, messages)
            except Exception as e:
                print(f"Error adding messages to database: {e}")
        else:
            print("messages not added to the database")
        self.history.append({"role": "user", "content": question})  # store raw question
        self.history.append({"role": "model", "content": answer})
        generation_time = time.time() - generation_start
        return answer, chunks, retrieval_time, generation_time

def call_llm(history: list[dict]) -> str:
    """
    Send conversation history + system instruction to Gemini, return the answer text.

    Args:
        history: list of {"role": "user"/"model", "content": "..."} turns
                 (NOT including the system message)
        system_message: the standing instructions for the assistant

    Returns:
        The model's text response as a plain string.
    """
    # Translate our internal role names into Gemini's expected format.
    contents = [
        {
            "role": turn["role"],
            "parts": [{"text": turn["content"]}],
        }
        for turn in history
    ]
    #payload is a JSON wrapper for the system message and the conversation history
    #so that the post request to the Gemini API can be made with the correct structure
    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_MESSAGE}]
        },
        "contents": contents,
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    """==============================OPENROUTER MODEL CALL=============================="""
    OPENROUTER_MESSAGES = [
        {"role": "system", "content": SYSTEM_MESSAGE},
    ]
    OPENROUTER_MESSAGES.extend([
        {"role": "user", "content": turn["content"]} 
        if turn["role"] == "user" 
        else {"role": "assistant", "content": turn["content"]}
        # {"role": turn["role"], "content": turn["content"]}  
        for turn in history
    ])
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_HEADERS = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    #letting the requests handle the json.dumps here to convert the python dictionary to a json string, so that it can be sent in the post request
    OPENROUTER_PAYLOAD = {
            "model": "minimax/minimax-m3:free",
            "messages": OPENROUTER_MESSAGES,
            "reasoning": {"enabled": True}
        }

    # Extract the assistant message with reasoning_details
    


    try:
        print("Calling Gemini API...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # raise RuntimeError(f"Gemini API call failedy: {e}")
        print (f"Gemini API call failed: {e}")
        print(f"\n\nGemini failed — falling back to OpenRouter...")
        try:
            response2 = requests.post(
                    url=OPENROUTER_API_URL,
                    headers=OPENROUTER_HEADERS,
                    #making the requests handle the json.dumps here to convert the python dictionary to a json string, so that it can be sent in the post request
                    json=OPENROUTER_PAYLOAD,
                    timeout=30
                )
        
            # status = response2.status_code #get the status of the response, if it is not 200, raise an error
            # if status != 200:
            #     # raise RuntimeError(f"OpenRouter API call failed: {response2}")
            #     print(f"OpenRouter API call failed: {response2}")
            #     return "all models are currently unavailable, please try again later."
            
            response2.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        except requests.exceptions.RequestException as e2:
            print(f"OpenRouter API call failed: {e2}")
            #if the error is a timeout retry, or a connection error, we can retry the request, otherwise we raise an error
            if isinstance(e2, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                print("Retrying OpenRouter API call...")
                try:
                    response2 = requests.post(
                        url=OPENROUTER_API_URL,
                        headers=OPENROUTER_HEADERS,
                        json=OPENROUTER_PAYLOAD,
                        timeout=30
                    )
                    response2.raise_for_status()
                except requests.exceptions.RequestException as e3:
                    print(f"OpenRouter API retry failed: {e3}")
                    return "all models are currently unavailable, please try again later."
            else:
                return "all models are currently unavailable, please try again later."
            
        response2 = response2.json()
        response2 = response2['choices'][0]['message']
        response2 = response2['content']
        return response2  # Return the OpenRouter response if Gemini fails

    print("Gemini API call successful.")
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {data}") from e

