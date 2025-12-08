from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm import llm

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str

@router.post("/urdu")
async def translate_urdu(req: TranslateRequest):

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prompt = f"""
Translate the following content into Urdu.
Keep formatting clean. Do NOT add anything extra, only translate the text.

Text:
{req.text}

Urdu Translation:
"""

    try:
        translation = llm.generate(prompt)
        return {"translation": translation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
