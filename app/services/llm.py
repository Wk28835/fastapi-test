import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY environment variable is missing")

genai.configure(api_key=GEMINI_API_KEY)

# Select your model
MODEL_NAME = "gemini-2.5-flash"

class LLMService:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)

            if hasattr(response, "text") and response.text:
                return response.text

            # fallback
            return "⚠️ Gemini returned an empty response."

        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"

# Singleton instance
llm = LLMService()
