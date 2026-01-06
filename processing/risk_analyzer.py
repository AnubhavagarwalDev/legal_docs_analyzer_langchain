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


def build_risk_prompt() -> PromptTemplate:
    template = """
You are a legal risk analysis assistant.

Task:
Assess whether the following legal clause poses potential risk to a non-lawyer user.

Rules:
- Do NOT give legal advice
- Do NOT claim illegality
- Focus on user impact and imbalance
- Be cautious and neutral
- Return ONLY valid JSON
- Do NOT include explanations or markdown

JSON format:
{{
  "risk_level": "low | medium | high",
  "risk_types": ["string"],
  "risk_summary": "string"
}}

Clause category:
{categories}

Clause text:
{text}

Detected red flags:
{red_flags}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["categories", "text", "red_flags"]
    )


def analyze_risk(
    clause: Dict[str, any],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0
) -> Dict[str, any]:
    """
    Analyze risk level of a single clause.
    """

    red_flags = detect_red_flags(clause.get("text", ""))

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_risk_prompt()

    response = llm.invoke(
        prompt.format(
            categories=", ".join(clause.get("categories", [])),
            text=clause.get("text", ""),
            red_flags=", ".join(red_flags) if red_flags else "None"
        )
    )

    parsed = safe_json_parse(response.content)

    return {
        "risk_level": parsed.get("risk_level", "low"),
        "risk_types": parsed.get("risk_types", []),
        "risk_summary": parsed.get("risk_summary", ""),
        "red_flags": red_flags
    }
