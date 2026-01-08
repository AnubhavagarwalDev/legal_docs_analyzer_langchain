from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from retrieval.embeddings import get_embedding_model
from utils.cache_utils import (
    compute_text_hash,
    get_cached_result,
    set_cached_result
)


def build_documents(clauses: List[Dict]) -> List[Document]:
    documents = []

    for clause in clauses:
        metadata = {
            "chunk_id": clause.get("chunk_id"),
            "heading": clause.get("heading"),
            "categories": clause.get("categories"),
            "risk_level": clause.get("risk", {}).get("risk_level", "low"),
        }

        documents.append(
            Document(
                page_content=clause.get("text", ""),
                metadata=metadata,
            )
        )

    return documents


def create_vectorstore(
    clauses: List[Dict],
    persist_path: str = "vectorstore/",
) -> FAISS:
    embedding_model = get_embedding_model()
    documents = build_documents(clauses)

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    vectorstore.save_local(persist_path)
    return vectorstore


def load_vectorstore(
    persist_path: str = "vectorstore/",
) -> FAISS:
    embedding_model = get_embedding_model()
    return FAISS.load_local(persist_path, embedding_model)
