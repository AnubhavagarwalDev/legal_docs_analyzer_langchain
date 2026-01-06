from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY
from utils.json_utils import safe_json_parse


def build_simplifier_prompt() -> PromptTemplate:
    template = """
You are a legal document simplification system.

Task:
Explain the following legal clause in clear, simple language for a non-lawyer.

Rules:
- Do NOT give legal advice
- Do NOT change the meaning
- Be neutral and factual
- Avoid legal jargon
- Keep it concise and understandable
- Return ONLY valid JSON
- Do NOT include explanations or markdown

JSON format:
{{
  "simple_explanation": "string",
  "user_impact": "string",
  "key_points": ["string"]
}}

Clause category:
{categories}

Clause text:
{text}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["categories", "text"]
    )


def simplify_clause(
    clause: Dict[str, any],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.2
) -> Dict[str, List[str]]:
    """
    Simplify a single legal clause into plain English.
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_simplifier_prompt()

    response = llm.invoke(
        prompt.format(
            categories=", ".join(clause.get("categories", [])),
            text=clause.get("text", "")
        )
    )

    parsed = safe_json_parse(response.content)

    return {
        "simple_explanation": parsed.get("simple_explanation", ""),
        "user_impact": parsed.get("user_impact", ""),
        "key_points": parsed.get("key_points", [])
    }
