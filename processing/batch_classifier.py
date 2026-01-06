from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY
from utils.json_utils import safe_json_parse


ALLOWED_CATEGORIES = [
    "obligations",
    "payments_fees",
    "termination",
    "penalties",
    "liability",
    "data_privacy",
    "restrictions",
    "dispute_resolution",
    "intellectual_property",
    "general"
]


def build_batch_classifier_prompt() -> PromptTemplate:
    template = """
You are a legal clause classification system.

Task:
For EACH clause below, classify it into one or more categories.

Allowed categories:
{allowed_categories}

Rules:
- Use ONLY the allowed categories
- Return ONLY valid JSON
- Do NOT include explanations, markdown, or extra text
- Output must be a JSON ARRAY
- Each array item corresponds to ONE clause
- If unsure, use "general"

JSON format (STRICT):
[
  {{
    "chunk_id": "string",
    "categories": ["string"],
    "confidence": number
  }}
]

Clauses:
{clauses}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["clauses"],
        partial_variables={
            "allowed_categories": ", ".join(ALLOWED_CATEGORIES)
        }
    )


def format_clauses_for_batch(clauses: List[Dict]) -> str:
    """
    Prepare clauses for batch prompt.
    """
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


def classify_clauses_batch(
    clauses: List[Dict],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0
) -> Dict[str, Dict]:
    """
    Batch classify multiple clauses in ONE Gemini call.

    Returns:
        Dict keyed by chunk_id → classification result
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_batch_classifier_prompt()
    formatted_clauses = format_clauses_for_batch(clauses)

    response = llm.invoke(
        prompt.format(clauses=formatted_clauses)
    )

    parsed = safe_json_parse(response.content)

    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array from batch classifier")

    results_by_id: Dict[str, Dict] = {}

    for item in parsed:
        if not isinstance(item, dict):
            continue

        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue

        raw_categories = item.get("categories", [])
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]

        categories = [
            c for c in raw_categories
            if c in ALLOWED_CATEGORIES
        ] or ["general"]

        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        results_by_id[chunk_id] = {
            "categories": categories,
            "confidence": confidence
        }

    # Backfill any missing clauses to avoid silent drops
    for clause in clauses:
        cid = clause.get("chunk_id")
        if cid and cid not in results_by_id:
            results_by_id[cid] = {
                "categories": ["general"],
                "confidence": 0.0
            }
    print("RAW MODEL RESPONSE:\n", response.content)

    return results_by_id
