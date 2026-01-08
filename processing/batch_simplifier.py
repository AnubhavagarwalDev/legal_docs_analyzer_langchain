from typing import Dict, List
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY
from utils.cache_utils import (
    compute_clause_hash,
    get_cached_result,
    set_cached_result
)
from utils.json_utils import safe_json_parse


def build_batch_simplifier_prompt() -> PromptTemplate:
    template = """
You are a legal document simplification system.

Task:
For EACH clause below, explain it in clear, simple language for a non-lawyer.

Rules:
- Do NOT give legal advice
- Do NOT change the meaning
- Be neutral and factual
- Avoid legal jargon
- Keep explanations concise
- Return ONLY valid JSON
- Do NOT include explanations or markdown
- Output must be a JSON array
- Each item must correspond to one clause

JSON format:
[
  {{
    "chunk_id": "string",
    "simple_explanation": "string",
    "user_impact": "string",
    "key_points": ["string"]
  }}
]

Clauses:
{clauses}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["clauses"]
    )


def format_clauses_for_batch(clauses: List[Dict]) -> str:
    """
    Prepare clauses for batch simplification prompt.
    """
    formatted = []

    for clause in clauses:
        formatted.append(
            f"""
Clause ID: {clause.get("chunk_id")}
Category: {", ".join(clause.get("categories", []))}
Text:
{clause.get("text", "")}
"""
        )

    return "\n\n".join(formatted)


SIMPLIFIER_PROMPT_VERSION = "batch_simplifier_v1"


def simplify_clauses_batch(
    clauses: List[Dict],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.2
) -> Dict[str, Dict]:
    """
    Batch simplify multiple clauses in ONE Gemini call.

    Returns:
        Dict keyed by chunk_id → simplification result
    """

    cache_keys_by_id: Dict[str, str] = {}
    results_by_id: Dict[str, Dict] = {}
    clauses_to_infer: List[Dict] = []

    for clause in clauses:
        cid = clause.get("chunk_id")
        cache_key = compute_clause_hash(
            clause,
            prompt_version=SIMPLIFIER_PROMPT_VERSION,
            model_name=model_name
        )
        cache_keys_by_id[cid] = cache_key
        cached = get_cached_result("simplification", cache_key)
        if cached:
            results_by_id[cid] = cached
        else:
            clauses_to_infer.append(clause)

    if not clauses_to_infer:
        return results_by_id

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_batch_simplifier_prompt()
    formatted_clauses = format_clauses_for_batch(clauses_to_infer)

    response = llm.invoke(
        prompt.format(clauses=formatted_clauses)
    )

    parsed = safe_json_parse(response.content)

    for item in parsed:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        result = {
            "simple_explanation": item.get("simple_explanation", ""),
            "user_impact": item.get("user_impact", ""),
            "key_points": item.get("key_points", [])
        }

        results_by_id[chunk_id] = result
        cache_key = cache_keys_by_id.get(chunk_id)
        if cache_key:
            set_cached_result("simplification", cache_key, result)

    # Backfill missing with empty structures
    for clause in clauses:
        cid = clause.get("chunk_id")
        if cid and cid not in results_by_id:
            fallback = {
                "simple_explanation": "",
                "user_impact": "",
                "key_points": []
            }
            results_by_id[cid] = fallback
            cache_key = cache_keys_by_id.get(cid)
            if cache_key:
                set_cached_result("simplification", cache_key, fallback)

    return results_by_id
