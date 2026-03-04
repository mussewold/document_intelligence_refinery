from pathlib import Path
from typing import List
from ..models.document_profile import DocumentProfile

from ..services.triage_services.artifact_loader import DocumentArtifacts
from ..services.triage_services.origin_detector import detect_pdf_origin
from ..services.triage_services.layout_detector import detect_layout_complexity
from ..services.triage_services.language_detector import extract_text_and_detect_language
from ..services.triage_services.domain_classifier import guess_domain
from ..services.triage_services.cost_estimator import estimate_extraction_cost

import asyncio


class TriageAgent:
    async def analyze(self, pdf_path: str) -> DocumentProfile:
        pdf_path = Path(pdf_path)
        artifacts = DocumentArtifacts(str(pdf_path))

        origin_type = await detect_pdf_origin(artifacts)
        layout_complexity, element_counts = await detect_layout_complexity(artifacts)
        full_text, lang_result = await extract_text_and_detect_language(artifacts)

        primary_language = lang_result["lang"]
        language_confidence = lang_result["score"]

        domain_hint = guess_domain(full_text)
        estimated_cost = estimate_extraction_cost(
            text_length=len(full_text),
            layout_complexity=layout_complexity,
            element_counts=element_counts,
        )

        return DocumentProfile(
            file_name=pdf_path.name,
            origin_type=origin_type,
            primary_language=primary_language,
            language_confidence=language_confidence,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            estimated_extraction_cost=estimated_cost,
            total_pages=element_counts.get("total_pages", 0),
            total_text_length=len(full_text),
            element_counts=element_counts,
        )

    async def analyze_batch(self, pdf_paths: List[str]) -> List[DocumentProfile]:
        tasks = [self.analyze(p) for p in pdf_paths]
        results = await asyncio.gather(*tasks)
        return results