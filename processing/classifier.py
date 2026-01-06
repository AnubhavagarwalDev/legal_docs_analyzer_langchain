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


def build_classifier_prompt() -> PromptTemplate:
    template = """
You are a legal clause classification system.

Task:
Classify the given legal clause into one or more categories.

Allowed categories:
{allowed_categories}

Rules:
- Use ONLY the allowed categories
- Return ONLY a valid JSON object
- Do NOT include explanations or markdown
- Multiple categories are allowed
- If unsure, use "general"

JSON format:
{{
  "categories": ["string"],
  "confidence": number
}}

Clause heading:
{heading}

Clause text:
{text}
"""

    return PromptTemplate(
        template=template.strip(),
        input_variables=["heading", "text"],
        partial_variables={
            "allowed_categories": ", ".join(ALLOWED_CATEGORIES)
        }
    )


def classify_clause(
    clause: Dict[str, str],
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.0
) -> Dict[str, List[str]]:
    """
    Classify a single legal clause using Gemini.
    """

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )

    prompt = build_classifier_prompt()

    response = llm.invoke(
        prompt.format(
            heading=clause.get("heading", ""),
            text=clause.get("text", "")
        )
    )

    parsed = safe_json_parse(response.content)

    categories = [
        c for c in parsed.get("categories", [])
        if c in ALLOWED_CATEGORIES
    ]

    if not categories:
        categories = ["general"]

    return {
        "categories": categories,
        "confidence": float(parsed.get("confidence", 0.0))
    }
