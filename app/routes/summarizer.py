from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import logging
import google.generativeai as genai

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Define request model
class SummarizeRequest(BaseModel):
    text: str
    bullets: bool = True  # Whether to return summary as bullet points

@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Build prompt
        bullet_instruction = "Summarize the text in concise bullet points." if req.bullets else "Summarize the text in a short paragraph."
        prompt = f"""
You are an educational summarizer agent.
{bullet_instruction}

Text:
{req.text}

Summary:
"""
        # Call Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # Extract answer
        if hasattr(response, "text") and response.text:
            summary = response.text
        elif hasattr(response, "candidates") and response.candidates:
            try:
                summary = response.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                summary = "Received an empty response from the AI model."
        else:
            summary = "Unexpected response format from AI model."

        return {"summary": summary}

    except Exception as e:
        logger.error(f"Gemini summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to summarize text: {str(e)}")
