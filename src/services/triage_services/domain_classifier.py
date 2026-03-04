def guess_domain(text):
    text_lower = text.lower()

    if any(w in text_lower for w in ["bank", "loan", "interest", "account"]):
        return "financial"
    elif any(w in text_lower for w in ["law", "contract", "court", "regulation"]):
        return "legal"
    elif any(w in text_lower for w in ["experiment", "algorithm", "protocol", "data"]):
        return "technical"
    elif any(w in text_lower for w in ["disease", "patient", "treatment", "clinical"]):
        return "medical"
    else:
        return "general"