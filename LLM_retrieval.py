import json
from pathlib import Path

from embedding_pipeline import load_model, embed_text
from pgvector_store import get_connection, get_table_stats
from rag_prompt import create_rag_prompt, format_chunks_as_context

from dotenv import load_dotenv
import os

import requests


import psycopg
# from sentence_transformers import SentenceTransformer


model = load_model()
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("GEMINI_API")
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
# 7.the user current and previous questions are associated with role user
# 8.the assistant previous answers are associated with role model"""

def search_similar_chunks(question: str, limit: int = 5):
    """Embed a question and ask Postgres for the nearest matching chunks using cosine distance."""
    try:
        # question_vector = model.encode(question, normalize_embeddings=True)
        question_vector = embed_text(model, question)
    except Exception as e:
        print(f"ERROR: Failed to load or encode with embedding model: {e}")
        raise

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT
                            chunk_id,
                            chapter,
                            section,
                            source,
                            chunk_text,
                            word_count,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM fitness_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (question_vector.tolist(), question_vector.tolist(), limit),
                    )
                    rows = cur.fetchall()
                except psycopg.Error as e:
                    print(f"ERROR: Database query failed: {e}")
                    raise

        return rows
    except psycopg.OperationalError as e:
        print(f"ERROR: Failed to connect to the database. Is the Postgres container running? Details: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to search similar chunks: {e}")
        raise

# class ConversationSession:
#     def __init__(self):
#         self.history = [{"role": "system", "content": SYSTEM_MESSAGE}]
#     def ask(self, question: str) -> str:
#         chunks = search_similar_chunks(question)
#         # context = format_chunks_as_context(chunks)
#         # user_turn = {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
#         messages_to_send = self.history + create_rag_prompt(question, chunks)
#         answer = call_llm(messages_to_send)
#         self.history.append({"role": "user", "content": question})  # store raw question
#         self.history.append({"role": "model", "content": answer})
#         return answer
MIN_THRESHOLD = 0.4
class ConversationSession:
    def __init__(self):
        self.history=[]
    def ask(self, question: str) -> str:
        chunks = search_similar_chunks(question)

        #skipping LLM call in case of low similarity to save requests, tokens and time.
        if chunks and chunks[0][-1] < MIN_THRESHOLD:
            answer = "I don't have enough information to answer this question based on the provided context."
            self.history.append({"role": "user", "content": question})  # store raw
            self.history.append({"role": "model", "content": answer})
            return answer
        context = format_chunks_as_context(chunks)
        user_turn = {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        messages_to_send = self.history + [user_turn]
        answer = call_llm(messages_to_send)
        self.history.append({"role": "user", "content": question})  # store raw question
        self.history.append({"role": "model", "content": answer})
        return answer

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
        "x-goog-api-key": API_KEY,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {data}") from e

# def call_llm(messages: list[dict]) -> str:
#     """
#     Call the LLM API with the provided messages and return the assistant's response.
    
#     Args:
#         messages: List of message dictionaries, each with 'role' and 'content'.
        
#     Returns:
#         The assistant's response as a string.
#     """
#     # Placeholder for actual LLM API call
#     # Replace this with your specific LLM API integration code
#     response = llm_api_call(messages)  # This function should be defined to interact with your LLM
#     return response

# #method that calls the llm api and returns the response using the api key in the .env file
# def llm_api_call(messages: list[dict]) -> str:
#     """
#     Call the LLM API with the provided messages and return the assistant's response.
    
#     Args:
#         messages: List of message dictionaries, each with 'role' and 'content'.
        
#     Returns:
#         The assistant's response as a string.
#     """
#     # import requests

#     url = "https://api.gemini.com/v1/llm"  # Replace with the actual LLM API endpoint
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "messages": messages
#     }

#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()  # Raise an error for bad responses
#         data = response.json()
#         return data.get("assistant_response", "")  # Adjust based on actual API response structure
#     except requests.RequestException as e:
#         print(f"ERROR: Failed to call LLM API: {e}")
#         raise

if __name__ == "__main__":
    # Test the complete RAG pipeline: retrieval + LLM answer generation
    try:
        print("=" * 60)
        print("FITNESS & WELLNESS RAG CHATBOT")
        print("=" * 60)
        print("Type your questions below. Type 'exit' or 'quit' to end.\n")
    
        session = ConversationSession()
    
        while True:
            user_input = input("You: ").strip()
        
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
        
            if not user_input:
                print("Please enter a question.\n")
                continue
        
            try:
                answer = session.ask(user_input)
                print(f"\nAssistant: {answer}\n")
            except Exception as e:
                print(f"\nError: {e}\n")
        
    except Exception as e:
        print(f"RAG pipeline failed: {e}")
        exit(1)