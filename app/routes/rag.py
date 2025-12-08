from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import logging
import google.generativeai as genai
from app.core.embeddings import embed_texts
from app.core.qdrant_db import client, COLLECTION

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

TOP_K = int(os.getenv("TOP_K", 4))

class Query(BaseModel):
    user_id: Optional[str] = None
    text: Optional[str] = None
    question: str
    only_selected: bool = False

@router.post("/query")
async def query(q: Query):
    try:
        # ========== VALIDATION =============
        if not q.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # ========== CASE 1: ONLY SELECTED TEXT =============
        if q.only_selected:
            if not q.text or not q.text.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="Text must be provided when only_selected is true"
                )
            contexts = [{"payload": {"text": q.text, "doc_path": "Selected Text"}}]
            logger.info(f"Using selected text only, length: {len(q.text)}")
        
        else:
            # ========== CASE 2: VECTOR SEARCH =============
            # Embed the user question
            try:
                q_emb = embed_texts([q.question])[0]
                logger.info(f"Question embedded successfully, vector length: {len(q_emb)}")
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to embed question")

            # Search Qdrant
            try:
                # IMPORTANT: query_points() returns different structures in different Qdrant versions
                # Let's handle both tuple and object responses
                search_result = client.query_points(
                    collection_name=COLLECTION,
                    query=q_emb,
                    limit=TOP_K
                )
                logger.info(f"Qdrant search completed, result type: {type(search_result)}")
                
                # Convert hits to a uniform format
                contexts = []
                if hasattr(search_result, 'points'):  # Object response
                    points = search_result.points
                    for point in points:
                        if hasattr(point, 'score') and hasattr(point, 'payload'):
                            contexts.append({
                                "score": point.score,
                                "payload": point.payload
                            })
                elif isinstance(search_result, tuple):  # Tuple response
                    # Handle tuple format (score, payload, id, etc.)
                    for hit in search_result:
                        if isinstance(hit, tuple) and len(hit) >= 2:
                            score, payload = hit[0], hit[1]
                            contexts.append({
                                "score": float(score) if score is not None else 0.0,
                                "payload": payload
                            })
                elif isinstance(search_result, list):  # List response
                    for hit in search_result:
                        if isinstance(hit, dict):  # Dict format
                            contexts.append({
                                "score": hit.get('score', 0.0),
                                "payload": hit.get('payload', {})
                            })
                        elif hasattr(hit, '__dict__'):  # Object with attributes
                            contexts.append({
                                "score": getattr(hit, 'score', 0.0),
                                "payload": getattr(hit, 'payload', {})
                            })
                
                logger.info(f"Extracted {len(contexts)} contexts from search results")
                
                if not contexts:
                    logger.warning("No contexts found in vector search")
                    # Fallback to empty context
                    contexts = [{"payload": {"text": "", "doc_path": "No results found"}}]
                    
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

        # ========== BUILD RAG CONTEXT =============
        prompt_parts = []
        sources = []
        
        for i, c in enumerate(contexts):
            # Safely extract payload
            payload = c.get("payload", {}) if isinstance(c, dict) else {}
            text = payload.get("text", "")
            doc_path = payload.get("doc_path", f"Source {i+1}")
            
            if text and text.strip():
                prompt_parts.append(
                    f"Source {i+1} (path: {doc_path}):\n{text}\n---\n"
                )
                sources.append({
                    "path": doc_path,
                    "snippet": text[:300] + "..." if len(text) > 300 else text,
                    "score": c.get("score", 0.0) if isinstance(c, dict) else 0.0
                })
        
        if not prompt_parts:
            prompt_parts = ["No context provided."]
            sources = [{"path": "No sources", "snippet": "No relevant documents found"}]
        
        context_text = "\n".join(prompt_parts)
        
        # ========== BUILD PROMPT =============
        system = (
               "You are an educational RAG assistant. "
                "You MUST answer ONLY using the provided context. "
                "If the answer is not in the context, respond with: "
                "'The context does not contain information about this question.' "
        )
        
        full_prompt = f"""
{system}

Context:
{context_text}

Question:
{q.question}

Answer:
"""
        logger.info(f"Prompt built, context length: {len(context_text)} chars")
        print("=== CONTEXT SENT TO GEMINI ===")
        print(context_text[:1000])
        # ========== GEMINI CALL =============
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(full_prompt)
            
            # Extract answer safely
            if hasattr(response, 'text') and response.text:
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    answer = response.candidates[0].content.parts[0].text
                except (AttributeError, IndexError):
                    answer = "I received an empty response from the AI model."
            else:
                answer = "I received an unexpected response format from the AI model."
                
            logger.info(f"Gemini response received, answer length: {len(answer)}")
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            answer = f"Error generating response: {str(e)}"

        # ========== RETURN RESPONSE =============
        return {
            "answer": answer,
            "sources": sources,
            "context_count": len(contexts)
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")