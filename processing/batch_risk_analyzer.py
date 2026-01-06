from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GEMINI_API_KEY
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

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_batch_risk_prompt()
    formatted_clauses = format_clauses_for_batch(clauses)

    response = llm.invoke(
        prompt.format(clauses=formatted_clauses)
    )

    parsed = safe_json_parse(response.content)

    results_by_id = {}

    for item in parsed:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue

        results_by_id[chunk_id] = {
            "risk_level": item.get("risk_level", "low"),
            "risk_types": item.get("risk_types", []),
            "risk_summary": item.get("risk_summary", ""),
            "red_flags": detect_red_flags(
                next(
                    c.get("text", "")
                    for c in clauses
                    if c.get("chunk_id") == chunk_id
                )
            )
        }

    return results_by_id
