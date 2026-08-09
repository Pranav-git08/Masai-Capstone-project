import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ensure support_assistant directory is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from support_assistant.graph import app_graph, AssistantResponse
from support_assistant.vectorstore import init_vectorstore

app = FastAPI(
    title="Zepto Support Assistant API",
    description="Module 3 RAG Assistant with LangGraph and FastAPI",
    version="1.0.0"
)

# Startup event to populate vector DB
@app.on_event("startup")
def startup_event():
    init_vectorstore()

class AskRequest(BaseModel):
    query: str = Field(..., example="What is the delivery fee for orders below 149?")

@app.post("/ask", response_model=AssistantResponse)
def ask_question(request: AskRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "response": None
    }
    
    final_state = app_graph.invoke(initial_state)
    return final_state["response"]

@app.get("/")
def root():
    return {"status": "ok", "message": "Zepto Support Assistant API is running"}
