from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq

app = FastAPI()

llm_obj = ChatGroq(model="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY"))


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
	return {"response":"Hello"}

# C:\Users\karth>python -m uvicorn p2:app --reload