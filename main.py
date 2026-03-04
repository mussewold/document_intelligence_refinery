import asyncio
from src.agents.triage import TriageAgent

async def main():
    agent = TriageAgent()
    pdf_files = [
        "./data/2013-E.C-Procurement-information.pdf",
        "./data/2013-E.C-Audit-finding-information.pdf"
    ]

    results = await agent.analyze_batch(pdf_files)

    for r in results:
        print(r.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())