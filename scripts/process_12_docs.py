import asyncio
import json
import random
from pathlib import Path

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.services.triage_services.artifact_loader import DocumentArtifacts
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.services.fact_table_extractor import FactTableExtractor
from src.agents.query_agent import RefineryAgent

DATA_DIR = Path("data")

async def process_docs():
    # Hardcode smallest 12 diverse PDFs to avoid huge CPU inference times
    pdf_names = [
        "2020_Audited_Financial_Statement_Report.pdf",
        "2021_Audited_Financial_Statement_Report.pdf",
        "2022_Audited_Financial_Statement_Report.pdf",
        "Consumer Price Index March 2025.pdf",
        "Consumer Price Index July 2025.pdf",
        "Consumer Price Index, April 2025.pdf",
        "2013-E.C-Audit-finding-information.pdf",
        "2013-E.C-Assigned-regular-budget-and-expense.pdf",
        "2013-E.C-Procurement-information.pdf",
        "ETHIO_RE_AT_A_GLANCE_2023_24.pdf",
        "tax_expenditure_ethiopia_2021_22.pdf",
        "CBE Annual Report 2012-13.pdf"
    ]
    selected = [DATA_DIR / name for name in pdf_names if (DATA_DIR / name).exists()]
    print(f"Selected {len(selected)} documents for processing.")

    processed_doc_ids = []
    
    triage = TriageAgent()
    router = ExtractionRouter()
    chunker = ChunkingEngine()
    pib = PageIndexBuilder()
    fact_ext = FactTableExtractor()

    for pdf_path in selected:
        print(f"\nProcessing {pdf_path.name}...")
        try:
            profile = await triage.analyze(str(pdf_path))
            artifacts = DocumentArtifacts(str(pdf_path))
            result = await router.extract(profile, artifacts, doc_id=pdf_path.name)
            doc = result.document

            ldus = chunker.build_ldus(doc)
            root = await pib.build(doc, ldus)
            fact_ext.extract_and_store(pdf_path.name, ldus)

            # Keep it around for queries
            processed_doc_ids.append((pdf_path.name, ldus, doc, root))
            print(f"  ✓ Success: {len(ldus)} LDUs, root pages: {root.page_start}-{root.page_end}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return processed_doc_ids

def run_qa(processed):
    agent = RefineryAgent()
    questions = [
        "What is the total revenue or income mentioned?",
        "What are the main risks or challenges?",
        "Provide a summary of the financial position or major findings."
    ]
    
    qa_results = {}
    
    for doc_id, ldus, doc, root in processed:
        qa_results[doc_id] = []
        print(f"\nRunning Q&A for {doc_id}...")
        for q in questions:
            try:
                res = agent.run(q, ldus=ldus, doc=doc, page_index_root=root)
                qa_results[doc_id].append({
                    "question": q,
                    "answer": res.answer,
                    "provenance": [c.model_dump() for c in res.provenance_chain]
                })
                print(f"  ✓ Answered: {q}")
            except Exception as e:
                print(f"  ✗ Error Q&A: {e}")
                import traceback
                traceback.print_exc()
                
    return qa_results

def save_markdown(qa_results):
    md_path = Path("12_QA_examples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Document Q&A Examples\n\n")
        
        for doc_id, items in qa_results.items():
            f.write(f"## Document: `{doc_id}`\n\n")
            for item in items:
                f.write(f"**Q:** {item['question']}\n\n")
                f.write(f"**A:** {item['answer']}\n\n")
                
                prov = item['provenance']
                if prov:
                    f.write("**Provenance Citations:**\n")
                    for p in prov:
                        tool = p.get('tool_used', 'unknown')
                        page = p.get('page_number', '?')
                        content = p.get('snippet', '').replace('\n', ' ')[:150]
                        f.write(f"- [{tool}] p.{page}: \"{content}...\"\n")
                else:
                    f.write("*No provenance available.*\n")
                f.write("\n---\n\n")
                
    print(f"\nSaved Q&A examples to {md_path.absolute()}")

async def main():
    docs = await process_docs()
    if docs:
        qa = run_qa(docs)
        save_markdown(qa)

if __name__ == "__main__":
    asyncio.run(main())
