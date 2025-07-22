REFUTING_KEYWORDS = [
    "false", "hoax", "debunked", "incorrect", "misleading",
    "scam", "conspiracy", "fake news", "unproven", "unsupported"
]


SUPPORTING_KEYWORDS = [
    "true", "fact check: true", "correct", "accurate", "verified"
]

def generate_verdict(evidence_list: list):
    """
    Analyzes a list of evidence items to generate a fact-check verdict.

    Args:
        evidence_list (list): A list of dictionaries, where each dict is a search result.

    Returns:
        A dictionary containing the final verdict, score, and influential evidence.
    """
    if not evidence_list:
        return {
            "verdict": "Unproven",
            "summary": "Could not find enough evidence to make a determination.",
            "score": 0,
            "influential_evidence": []
        }

    score = 0
    influential_evidence = []

    # Analyze each piece of evidence
    for evidence in evidence_list:
        text_to_analyze = (evidence.get("title", "") + " " + evidence.get("snippet", "")).lower()
        
        found_refuting = any(keyword in text_to_analyze for keyword in REFUTING_KEYWORDS)
        found_supporting = any(keyword in text_to_analyze for keyword in SUPPORTING_KEYWORDS)

        # Simple scoring: refuting evidence has a stronger impact
        if found_refuting:
            score -= 1
            influential_evidence.append(evidence)
        elif found_supporting:
            score += 1
            influential_evidence.append(evidence)

    # Determine the final verdict based on the aggregate score
    if score <= -2:
        verdict = "Likely False"
        summary = "Multiple sources suggest this claim is inaccurate or misleading."
    elif score == -1:
        verdict = "Potentially Misleading"
        summary = "There is some evidence refuting this claim or suggesting it lacks context."
    elif score >= 2:
        verdict = "Likely True"
        summary = "Multiple sources appear to support this claim."
    elif score == 1:
        verdict = "Potentially True"
        summary = "Some evidence suggests this claim is accurate."
    else:
        verdict = "Contested / Unproven"
        summary = "Evidence is mixed or insufficient to form a strong conclusion."

    return {
        "verdict": verdict,
        "summary": summary,
        "score": score,
        "influential_evidence": influential_evidence[:3]
    }