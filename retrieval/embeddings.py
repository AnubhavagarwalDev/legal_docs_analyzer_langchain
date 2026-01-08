from langchain_huggingface import HuggingFaceEmbeddings

# Default local embedding model for semantic search
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """
    Returns a local HuggingFace embedding model for semantic retrieval.

    This uses all-MiniLM-L6-v2:
    - 384-dimensional embeddings
    - Fast and light, good for clause-level similarity
    - Runs locally (no external API calls)
    """
    return HuggingFaceEmbeddings(model_name=model_name)
