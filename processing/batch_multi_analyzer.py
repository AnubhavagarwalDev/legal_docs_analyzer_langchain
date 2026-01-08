from typing import Any, Dict, List
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY
from utils.cache_utils import (
    compute_clause_hash,
    get_cached_result,
    set_cached_result
)
from utils.json_utils import safe_json_parse
from processing.batch_classifier import ALLOWED_CATEGORIES

# Keep this in sync with prompt changes to invalidate cache safely.
MULTI_PROMPT_VERSION = "batch_multi_v1"


def build_batch_multi_prompt() -> PromptTemplate:
    template = """
You are a legal clause analyst.

Task:
For EACH clause below, perform ALL of these in one pass:
1) Classify into allowed categories.
2) Provide a plain-English explanation.
3) Assess risk level and risk summary.

Allowed categories: obligations, payments_fees, termination, penalties,
liability, data_privacy, restrictions, dispute_resolution,
intellectual_property, general

Rules:
- Return ONLY valid JSON.
- NO markdown, headers, commentary, or extra text.
- Output must be a JSON array; one item per clause.
- Keep explanations concise and neutral; no legal advice.
- risk_level must be one of: low, medium, high.
- key_points should be short bullet phrases (no sentences).

STRICT JSON schema:
[
  {
    "chunk_id": "string",
    "categories": ["string"],
    "confidence": number,
    "simple_explanation": "string",
    "user_impact": "string",
    "key_points": ["string"],
    "risk_level": "low | medium | high",
    "risk_types": ["string"],
    "risk_summary": "string",
    "red_flags": ["string"]
  }
]

Clauses:
{clauses}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["clauses"]
    )


def format_clauses_for_batch(clauses: List[Dict]) -> str:
    formatted = []

    for clause in clauses:
        formatted.append(
            f"""
Clause ID: {clause.get("chunk_id")}
Heading: {clause.get("heading", "")}
Text:
{clause.get("text", "")}
"""
        )

    return "\n\n".join(formatted)


def analyze_clauses_batch(
    clauses: List[Dict],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0
) -> Dict[str, Dict[str, Any]]:
    """
    Single-shot batch that returns classification, simplification,
    and risk analysis for each clause in one Gemini call.
    Uses a hash-based cache to avoid re-processing unchanged clauses.
    """
    cache_keys_by_id: Dict[str, str] = {}
    results_by_id: Dict[str, Dict[str, Any]] = {}
    clauses_to_infer: List[Dict] = []

    for clause in clauses:
        cid = clause.get("chunk_id")
        cache_key = compute_clause_hash(
            clause,
            prompt_version=MULTI_PROMPT_VERSION,
            model_name=model_name
        )
        cache_keys_by_id[cid] = cache_key
        cached = get_cached_result("multi_analysis", cache_key)
        if cached:
            results_by_id[cid] = cached
        else:
            clauses_to_infer.append(clause)

    if clauses_to_infer:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY
        )

        prompt = build_batch_multi_prompt()
        formatted_clauses = format_clauses_for_batch(clauses_to_infer)

        response = llm.invoke(
            prompt.format(clauses=formatted_clauses)
        )

        parsed = safe_json_parse(response.content)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array from multi analyzer")

        for item in parsed:
            if not isinstance(item, dict):
                continue

            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            # Normalize fields with defaults
            raw_categories = item.get("categories", [])
            if isinstance(raw_categories, str):
                raw_categories = [raw_categories]

            categories = [
                c for c in raw_categories if c in ALLOWED_CATEGORIES
            ] or ["general"]

            confidence = item.get("confidence", 0.0) or 0.0
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            result = {
                "categories": categories,
                "confidence": confidence,
                "simple_explanation": item.get("simple_explanation", ""),
                "user_impact": item.get("user_impact", ""),
                "key_points": item.get("key_points", []),
                "risk_level": item.get("risk_level", "low"),
                "risk_types": item.get("risk_types", []),
                "risk_summary": item.get("risk_summary", ""),
                "red_flags": item.get("red_flags", [])
            }

            results_by_id[chunk_id] = result
            cache_key = cache_keys_by_id.get(chunk_id)
            if cache_key:
                set_cached_result("multi_analysis", cache_key, result)

    # Backfill any missing clauses to avoid silent drops
    for clause in clauses:
        cid = clause.get("chunk_id")
        if cid and cid not in results_by_id:
            fallback = {
                "categories": ["general"],
                "confidence": 0.0,
                "simple_explanation": "",
                "user_impact": "",
                "key_points": [],
                "risk_level": "low",
                "risk_types": [],
                "risk_summary": "",
                "red_flags": []
            }
            results_by_id[cid] = fallback
            cache_key = cache_keys_by_id.get(cid)
            if cache_key:
                set_cached_result("multi_analysis", cache_key, fallback)

    return results_by_id
