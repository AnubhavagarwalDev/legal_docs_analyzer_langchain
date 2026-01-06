from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GEMINI_API_KEY


def get_embedding_model(
    model_name: str = "models/embedding-001"
) -> GoogleGenerativeAIEmbeddings:
    """
    Returns a Gemini embedding model for semantic retrieval.
    """
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_document"
    )
