from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.rag import router as rag_router
from app.routes.summarizer import router as summarizer_router
from app.routes.qa import router as qa_router
from app.routes.translate import router as translate_router


app = FastAPI(title="Panaversity RAG Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Docusaurus dev
        "https://hackathonq4-seven.vercel.app",
        "https://fastapi-test-murex-sigma.vercel.app",  # (add later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/")
def root():
   return {"status": "RAG API running successfully"}

# RAG endpoints
app.include_router(rag_router, prefix="/rag")
# Summarizer endpoints
app.include_router(summarizer_router, prefix="/summarizer", tags=["Summarizer"])

app.include_router(qa_router, prefix="/qa")
app.include_router(translate_router, prefix="/translate")
