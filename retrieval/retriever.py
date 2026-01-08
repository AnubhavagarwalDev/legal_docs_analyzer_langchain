from typing import List, Dict
from langchain_community.vectorstores import FAISS


def retrieve_clauses(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    filter_categories: List[str] = None,
    filter_risk_level: str = None
) -> List[Dict]:
    """
    Retrieve relevant clause chunks using semantic search.
    """

    docs = vectorstore.similarity_search(query, k=k)

    results = []

    for doc in docs:
        metadata = doc.metadata

        # Category filter
        if filter_categories:
            if not set(filter_categories).intersection(set(metadata.get("categories", []))):
                continue

        # Risk level filter
        if filter_risk_level:
            if metadata.get("risk_level") != filter_risk_level:
                continue

        results.append({
            "text": doc.page_content,
            "metadata": metadata
        })

    return results
