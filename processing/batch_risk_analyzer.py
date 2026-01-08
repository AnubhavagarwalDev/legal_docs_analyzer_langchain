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


RED_FLAG_PHRASES = [
    "non-refundable",
    "without notice",
    "sole discretion",
    "no liability",
    "not liable",
    "waive",
    "indemnify",
    "penalty",
    "termination at any time"
]


def detect_red_flags(text: str) -> List[str]:
    text_lower = text.lower()
    return [phrase for phrase in RED_FLAG_PHRASES if phrase in text_lower]


def build_batch_risk_prompt() -> PromptTemplate:
    template = """
You are a legal risk analysis assistant.

Task:
For EACH clause below, assess whether it poses potential risk to a non-lawyer user.

Rules:
- Do NOT give legal advice
- Do NOT claim illegality
- Focus on user impact and imbalance
- Be cautious and neutral
- Return ONLY valid JSON
- Do NOT include explanations or markdown
- Output must be a JSON array
- Each item must correspond to one clause

JSON format:
[
  {{
    "chunk_id": "string",
    "risk_level": "low | medium | high",
    "risk_types": ["string"],
    "risk_summary": "string"
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
    Prepare clauses for batch risk analysis prompt.
    """
    formatted = []

    for clause in clauses:
        red_flags = detect_red_flags(clause.get("text", ""))

        formatted.append(
            f"""
Clause ID: {clause.get("chunk_id")}
Category: {", ".join(clause.get("categories", []))}
Detected red flags: {", ".join(red_flags) if red_flags else "None"}

Text:
{clause.get("text", "")}
"""
        )

    return "\n\n".join(formatted)


RISK_PROMPT_VERSION = "batch_risk_v1"


def analyze_risks_batch(
    clauses: List[Dict],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0
) -> Dict[str, Dict]:
    """
    Batch analyze risk for multiple clauses in ONE Gemini call.

    Returns:
        Dict keyed by chunk_id → risk analysis result
    """

    cache_keys_by_id: Dict[str, str] = {}
    results_by_id: Dict[str, Dict] = {}
    clauses_to_infer: List[Dict] = []

    for clause in clauses:
        cid = clause.get("chunk_id")
        cache_key = compute_clause_hash(
            clause,
            prompt_version=RISK_PROMPT_VERSION,
            model_name=model_name
        )
        cache_keys_by_id[cid] = cache_key
        cached = get_cached_result("risk", cache_key)
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

    prompt = build_batch_risk_prompt()
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
            "risk_level": item.get("risk_level", "low"),
            "risk_types": item.get("risk_types", []),
            "risk_summary": item.get("risk_summary", ""),
            "red_flags": detect_red_flags(
                next(
                    (
                        c.get("text", "")
                        for c in clauses
                        if c.get("chunk_id") == chunk_id
                    ),
                    ""
                )
            )
        }

        results_by_id[chunk_id] = result
        cache_key = cache_keys_by_id.get(chunk_id)
        if cache_key:
            set_cached_result("risk", cache_key, result)

    for clause in clauses:
        cid = clause.get("chunk_id")
        if cid and cid not in results_by_id:
            fallback = {
                "risk_level": "low",
                "risk_types": [],
                "risk_summary": "",
                "red_flags": detect_red_flags(clause.get("text", ""))
            }
            results_by_id[cid] = fallback
            cache_key = cache_keys_by_id.get(cid)
            if cache_key:
                set_cached_result("risk", cache_key, fallback)

    return results_by_id
