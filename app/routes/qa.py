from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import llm  # your existing LLM class

router = APIRouter()

class QARequest(BaseModel):
    question: str

@router.post("/answer")
async def answer_question(payload: QARequest):
    try:
        response = llm.generate(
            prompt=f"Answer the question concisely:\n\n{payload.question}"
        )
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
