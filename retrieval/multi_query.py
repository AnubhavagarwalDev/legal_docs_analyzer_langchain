from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY

from retrieval.retriever import retrieve_clauses


def generate_alternative_queries(
    query: str,
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0,
    num_queries: int = 4
) -> List[str]:
    """
    Generate alternative semantic formulations of a user query.
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    template = """
You are rewriting a user question about a legal document.

Task:
Generate {num_queries} alternative queries that express the SAME intent,
using formal or legal-style wording.

Rules:
- Do NOT add new intent
- Do NOT answer the question
- Keep each query short
- Return one query per line
- Do NOT add explanations

User question:
{query}
"""

    prompt = PromptTemplate(
        template=template.strip(),
        input_variables=["query", "num_queries"]
    )

    response = llm.invoke(
        prompt.format(query=query, num_queries=num_queries)
    )

    queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    return queries[:num_queries]


def multi_query_retrieve(
    vectorstore,
    user_query: str,
    k_per_query: int = 3,
    filter_categories: List[str] = None,
    filter_risk_level: str = None
) -> List[Dict]:
    """
    Perform multi-query semantic retrieval and merge results.
    """

    queries = [user_query] + generate_alternative_queries(user_query)

    seen = set()
    merged_results = []

    for q in queries:
        results = retrieve_clauses(
            vectorstore,
            query=q,
            k=k_per_query,
            filter_categories=filter_categories,
            filter_risk_level=filter_risk_level
        )

        for r in results:
            clause_id = r["metadata"].get("chunk_id")
            if clause_id not in seen:
                seen.add(clause_id)
                merged_results.append(r)

    return merged_results
