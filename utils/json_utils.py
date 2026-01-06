import json
import re

def safe_json_parse(text: str):
    """
    Extracts the FIRST valid JSON object or array from LLM output.
    Raises ValueError if none found.
    """

    # Remove markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try full parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex to extract first JSON object or array
    match = re.search(
        r"(\{.*?\}|\[.*?\])",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError("No JSON object found in response")

    json_str = match.group(1)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON extracted: {e}")
