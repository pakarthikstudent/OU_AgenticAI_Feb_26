from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama

app = FastAPI()

# Initialize Ollama LLM  
llm_obj = ChatOllama(
    model="gemma2:2b"
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_llm(query: Query):
    response = llm_obj.invoke(query.question)
    return {
        "question": query.question,
        "answer": response.content
    }

@app.get("/")
def f1():
    return {"response": "Hello"}
