from fastapi import FastAPI
from pydantic import BaseModel
from data_retrival import ask_question
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Resume AI API is running"}

@app.post("/ask")
def ask(req: QueryRequest):
    answer = ask_question(req.question)
    return {
        "question": req.question,
        "answer": answer
    }