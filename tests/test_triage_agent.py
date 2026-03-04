import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile

@pytest.mark.asyncio
async def test_analyze_success():
    # Setup mocks
    with patch("src.agents.triage.DocumentArtifacts") as MockArtifacts, \
         patch("src.agents.triage.detect_pdf_origin", new_callable=AsyncMock) as mock_origin, \
         patch("src.agents.triage.detect_layout_complexity", new_callable=AsyncMock) as mock_layout, \
         patch("src.agents.triage.extract_text_and_detect_language", new_callable=AsyncMock) as mock_lang, \
         patch("src.agents.triage.guess_domain") as mock_domain, \
         patch("src.agents.triage.estimate_extraction_cost") as mock_cost:
        
        # Configure return values
        mock_origin.return_value = "native_digital"
        mock_layout.return_value = ("single_column", {"text": 10, "total_pages": 1})
        mock_lang.return_value = ("Sample text", {"lang": "en", "score": 0.99})
        mock_domain.return_value = "legal"
        mock_cost.return_value = 0.05
        
        agent = TriageAgent()
        profile = await agent.analyze("dummy.pdf")
        
        # Assertions
        assert profile.file_name == "dummy.pdf"
        assert profile.origin_type == "native_digital"
        assert profile.primary_language == "en"
        assert profile.domain_hint == "legal"
        assert profile.layout_complexity == "single_column"
        assert profile.estimated_extraction_cost == 0.05
        assert profile.total_pages == 1
        assert profile.language_confidence == 0.99
