import re

def validate_query(query):
    """
    Ensures user queries are financial in nature and not harmful.
    """
    restricted_patterns = [
        r"\b(weapon|illegal|hack|scam|fraud)\b",  # Block inappropriate queries
        r"capital of .*",  # Block irrelevant geography questions
        r"who is .*",  # Block non-financial questions
    ]

    for pattern in restricted_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "This query is not allowed. Please ask financial-related questions."

    return True, "Valid query."

def filter_response(answer, confidence):
    """
    If confidence is too low, prevent hallucinated or misleading responses.
    """
    if confidence < 0.4:
        return "⚠️ The retrieved information is unreliable. Please verify with official sources."
    return answer
