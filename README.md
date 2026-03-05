# Document Intelligence Refinery

A production-oriented, async-first **Document Intelligence Pipeline**
for analyzing, triaging, and extracting structured content from PDF
documents.

The system classifies documents by: - Origin Type (native digital,
scanned, mixed, form-fillable) - Layout Complexity (single column,
multi-column, table-heavy, etc.) - Language - Domain Hint (financial,
legal, medical, etc.) - Estimated Extraction Cost

It then routes the document to the most cost-efficient extraction
strategy using a confidence-gated escalation mechanism.

------------------------------------------------------------------------

# Architecture Overview

The system is composed of two major phases:

## 1. Triage Phase (Profiling)

The `TriageAgent` analyzes a document and generates a `DocumentProfile`.

Signals include: - PDF origin detection - Layout complexity detection
(DocLayout-YOLO) - OCR + language detection - Domain classification -
Cost estimation

Output:

    .refinery/profiles/<filename>.json

------------------------------------------------------------------------

## 2. Extraction Phase (Strategy Routing)

The `ExtractionRouter` selects among three strategies:

### Strategy A --- Fast Text (Low Cost)

-   Uses pdfplumber
-   Designed for native digital + single column documents
-   Confidence scoring per page
-   Escalates if below threshold

### Strategy B --- Layout-Aware (Docling)

-   Used for multi-column or table-heavy documents
-   Produces structured tables, figures, and text blocks

### Strategy C --- Vision-Augmented (VLM via OpenRouter)

-   Used for scanned documents
-   Budget-guarded token usage
-   Supports Gemini Flash and GPT-4o-mini

Each extraction run is logged in:

    .refinery/extraction_ledger.jsonl

------------------------------------------------------------------------

# Project Structure

    document-intelligence-refinery/
    │
    ├── main.py
    ├── extraction_rules.yaml
    ├── pyproject.toml
    │
    ├── src/
    │   ├── agents/
    │   │   ├── triage.py
    │   │   └── extractor.py
    │   │
    │   ├── models/
    │   │   ├── document_profile.py
    │   │   ├── extracted_document.py
    │   │   ├── layout_extractor.py
    │   │   └── provenance*.py
    │   │
    │   ├── services/
    │   │   ├── extraction_ledger.py
    │   │   └── triage_services/
    │   │
    │   └── strategies/
    │       ├── fast_text.py
    │       ├── layout_extractor.py
    │       └── vision_extractor.py
    │
    └── .refinery/

------------------------------------------------------------------------

# Installation

## Requirements

-   Python 3.12+
-   Tesseract OCR installed
-   Poppler (for pdf2image)

## Install Dependencies

``` bash
pip install -e .
```

Or:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

## 1️⃣ Run Triage Only

``` python
from src.agents.triage import TriageAgent
import asyncio

async def run():
    agent = TriageAgent()
    profile = await agent.analyze("sample.pdf")
    print(profile.model_dump_json(indent=2))

asyncio.run(run())
```

------------------------------------------------------------------------

## 2️⃣ Run Extraction Router

``` python
from src.agents.extractor import ExtractionRouter

router = ExtractionRouter()
result = await router.extract(profile, artifacts)
```

------------------------------------------------------------------------

## 3️⃣ Batch Mode

``` python
async def main():
    agent = TriageAgent()
    pdf_files = ["doc1.pdf", "doc2.pdf"]
    results = await agent.analyze_batch(pdf_files)
```

------------------------------------------------------------------------

# DocumentProfile Schema

``` json
{
  "file_name": "example.pdf",
  "origin_type": "native_digital",
  "primary_language": "en",
  "layout_complexity": "single_column",
  "domain_hint": "financial",
  "estimated_extraction_cost": "fast_text_sufficient"
}
```

------------------------------------------------------------------------

# ExtractedDocument Schema

The unified internal format for all extractors.

Includes: - TextBlocks (with bounding boxes) - Tables (structured
cells) - Figures - Reading Order - Metadata (confidence, cost, engine)

------------------------------------------------------------------------

# Strategy Escalation Logic

1.  Try Fast Text (if applicable)
2.  If confidence \< threshold → escalate to Docling
3.  If layout extraction insufficient → escalate to Vision
4.  If scanned image → Vision directly

------------------------------------------------------------------------

# Confidence Model (Fast Text)

Signals: - Character count - Character density - Image area ratio - Font
metadata presence

Weighted combination produces a score in \[0,1\].

------------------------------------------------------------------------

# Cost Estimation Heuristic

  Condition                     Result
  ----------------------------- ----------------------
  Very short text               needs_vision_model
  Many figures                  needs_vision_model
  Multi-column or table-heavy   needs_layout_model
  Otherwise                     fast_text_sufficient

------------------------------------------------------------------------

# Environment Variables

For Vision Extraction:

    OPENROUTER_API_KEY=<your_key>

------------------------------------------------------------------------

# Testing

Run tests:

``` bash
pytest
```

------------------------------------------------------------------------

# Async Design

-   Heavy operations run in thread pool executors
-   Batch triage supported
-   Non-blocking OCR & layout detection

------------------------------------------------------------------------

# Extending the System

You can:

-   Add new extraction strategies
-   Replace cost estimator with ML model
-   Swap layout detector
-   Add document embedding index
-   Add API wrapper (FastAPI)

------------------------------------------------------------------------

# Production Considerations

-   Add caching layer
-   Add structured logging
-   Add monitoring around token usage
-   Persist extraction artifacts in database
-   Add retry strategy for VLM calls

------------------------------------------------------------------------

# License

MIT License

------------------------------------------------------------------------

# Author

Document Intelligence Refinery --- Advanced Async Document Processing
Framework
