"""
API contract 
GET /health
    returns { "status" : "ok"}

POST  /api/sessions
    create a new conversation session
    returns { "session_id" : "..."}

GET /api/sessions/{session_id}
    returns { "session_id" : "...", "messages" : [ { "role" : "user|model", "content" : "..." }, ... ] }

POST /api/sessions/{session_id}/messages
    ask a question
    request: {"question": "..."}
    returns: 
    {
        "answer": "...",
        "chunks": "[...]",
        "retrieval_time": 0.0,
        "generation_time": 0.0
    }
    
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pgvector_store import get_session_history, get_sessions, new_session
from pydantic import BaseModel
from LLM_client import ConversationSession


app = FastAPI(title="Fitness Chat API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


#health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


class MessageRequest(BaseModel):
    question:str

@app.post("/api/sessions")
async def create_session():
    """Create a new conversation session."""
    try:
        session_id = new_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/messages")
def add_message(session_id: str, message: MessageRequest):
    """ask a question."""
    try:
        session = ConversationSession(session_id)
        answer, chunks, retrieval_time, generation_time =  session.ask(message.question)
        return {
            "answer" : answer,
            "chunks" : chunks,
            "retrieval_time" : retrieval_time,
            "generation_time" : generation_time,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve the conversation history for a given session."""
    try:
        history = get_session_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/")
async def list_sessions():
    """get session IDs for all sessions."""
    try:
        session_IDs = get_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    sessions= []
    for session_id in session_IDs:
        """retrieve the conversation history for a each session"""
        try:
            messages = get_session_history(session_id)
            sessions.append({"session_id": session_id, "messages": messages})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    """returning all sessions with their messages"""
    return {"sessions": sessions}

@app.get("/api/session_ids")
async def get_session_ids():
    """Retrieve all session IDs."""
    try:
        session_IDs = get_sessions()
        return {"session_ids": session_IDs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


