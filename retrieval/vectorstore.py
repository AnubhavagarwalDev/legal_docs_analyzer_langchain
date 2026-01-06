from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.schema import Document

from retrieval.embeddings import get_embedding_model


def build_documents(clauses: List[Dict]) -> List[Document]:
    """
    Convert clause chunks into LangChain Document objects.
    """
    documents = []

    for clause in clauses:
        metadata = {
            "chunk_id": clause.get("chunk_id"),
            "heading": clause.get("heading"),
            "categories": clause.get("categories"),
            "risk_level": clause.get("risk", {}).get("risk_level", "low")
        }

        documents.append(
            Document(
                page_content=clause.get("text", ""),
                metadata=metadata
            )
        )

    return documents


def create_vectorstore(
    clauses: List[Dict],
    persist_path: str = "vectorstore/"
) -> FAISS:
    """
    Create and persist a FAISS vector store from clause chunks.
    """
    embedding_model = get_embedding_model()
    documents = build_documents(clauses)

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    vectorstore.save_local(persist_path)
    return vectorstore


def load_vectorstore(
    persist_path: str = "vectorstore/"
) -> FAISS:
    """
    Load an existing FAISS vector store.
    """
    embedding_model = get_embedding_model()
    return FAISS.load_local(persist_path, embedding_model)
