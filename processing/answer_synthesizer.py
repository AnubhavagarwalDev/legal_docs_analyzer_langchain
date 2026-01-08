from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY


def build_answer_prompt() -> PromptTemplate:
    """
    Prompt for grounded answer synthesis using retrieved clauses only.
    """

    template = """
You are answering a user's question about a legal document.

STRICT RULES:
- Answer ONLY using the provided clauses
- Do NOT add assumptions or external knowledge
- Do NOT give legal advice
- Be neutral and factual
- Do NOT include any preamble or commentary outside the answer format
- If the answer is not found in the clauses, clearly say so
- Cite clauses using clause ID and heading

User question:
{question}

Relevant clauses:
{clauses}

Answer format:
1. Direct answer in plain English
2. Supporting clauses (bullet points with clause ID and heading)
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["question", "clauses"]
    )


def format_clauses_for_prompt(retrieved_clauses: List[Dict]) -> str:
    """
    Format retrieved clauses into a readable block for the LLM.
    """

    formatted = []

    for clause in retrieved_clauses:
        meta = clause.get("metadata", {})
        formatted.append(
            f"""
Clause ID: {meta.get("chunk_id")}
Heading: {meta.get("heading")}
Category: {", ".join(meta.get("categories", []))}
Risk level: {meta.get("risk_level")}

Clause text:
{clause.get("text")}
"""
        )

    return "\n\n".join(formatted)


def synthesize_answer(
    question: str,
    retrieved_clauses: List[Dict],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.2
) -> Dict[str, str]:
    """
    Generate a grounded answer using retrieved clauses only.
    """

    if not retrieved_clauses:
        return {
            "answer": "The document does not contain enough information to answer this question."
        }

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_answer_prompt()
    formatted_clauses = format_clauses_for_prompt(retrieved_clauses)

    response = llm.invoke(
        prompt.format(
            question=question,
            clauses=formatted_clauses
        )
    )

    return {
        "answer": response.content.strip()
    }
