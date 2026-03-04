def estimate_extraction_cost(text_length, layout_complexity, element_counts):
    if text_length < 300:
        return "needs_vision_model"

    if element_counts.get("figure", 0) > element_counts.get("text", 0):
        return "needs_vision_model"

    if layout_complexity in ["multi_column", "table_heavy"]:
        return "needs_layout_model"

    return "fast_text_sufficient"