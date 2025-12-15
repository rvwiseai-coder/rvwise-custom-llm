"""
Configuration module for LlamaCloud RAG Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the RAG chatbot"""
    
    # LlamaCloud settings
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "llx-uv7xL8yfP4JtLA5xSGvoeasBoXOrsaTFDkQJWcRUip568ymL")
    LLAMA_CLOUD_ORG_ID = os.getenv("LLAMA_CLOUD_ORG_ID", "88219eba-28f6-4951-879b-75a417ca04c9")
    LLAMA_CLOUD_INDEX_NAME = os.getenv("LLAMA_CLOUD_INDEX_NAME", "rag-full-db-2025-09-05")
    LLAMA_CLOUD_PROJECT_NAME = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")
    
    # OpenAI settings (optional, if needed)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Streamlit settings
    APP_TITLE = "RVwise RAG Chatbot"
    APP_ICON = "🤖"
    
    # Retrieval settings
    TOP_K_RESULTS = 5  # Number of documents to retrieve

    # Server settings
    PORT = int(os.getenv("PORT", 8080))
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.LLAMA_CLOUD_API_KEY or cls.LLAMA_CLOUD_API_KEY == "llx-...":
            raise ValueError(
                "Please set your LLAMA_CLOUD_API_KEY in the environment variables or .env file"
            )
        return True
