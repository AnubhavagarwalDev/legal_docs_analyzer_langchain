from typing import Dict, List
from collections import Counter


RISK_WEIGHTS = {
    "low": 1,
    "medium": 3,
    "high": 6
}


def compute_risk_score(clauses: List[Dict]) -> int:
    total_score = 0
    max_possible = len(clauses) * RISK_WEIGHTS["high"]

    for clause in clauses:
        level = clause.get("risk", {}).get("risk_level", "low")
        total_score += RISK_WEIGHTS.get(level, 1)

    if max_possible == 0:
        return 0

    return int((total_score / max_possible) * 100)


def aggregate_document(clauses: List[Dict]) -> Dict:
    """
    Aggregate clause-level analysis into document-level insights.
    """

    risk_levels = Counter()
    category_distribution = Counter()
    risky_clauses = []

    for clause in clauses:
        # Risk stats
        risk = clause.get("risk", {})
        risk_level = risk.get("risk_level", "low")
        risk_levels[risk_level] += 1

        # Categories
        for cat in clause.get("categories", []):
            category_distribution[cat] += 1

        # Collect high-risk clauses
        if risk_level == "high":
            risky_clauses.append({
                "chunk_id": clause.get("chunk_id"),
                "heading": clause.get("heading"),
                "risk_summary": risk.get("risk_summary"),
                "red_flags": risk.get("red_flags", [])
            })

    risk_score = compute_risk_score(clauses)

    # Determine overall risk level
    if risk_score >= 70:
        overall_risk = "high"
    elif risk_score >= 40:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "overall_risk_level": overall_risk,
        "risk_score": risk_score,
        "risk_breakdown": dict(risk_levels),
        "category_distribution": dict(category_distribution),
        "top_risky_clauses": risky_clauses[:5]
    }
