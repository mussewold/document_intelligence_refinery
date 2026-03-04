import pytesseract
from fast_langdetect import detect
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()


async def ocr_page(page_np):
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(executor, lambda: pytesseract.image_to_string(page_np, lang="amh+eng"))
    return text


async def extract_text_and_detect_language(artifacts):
    images = await artifacts.load_images()
    tasks = [ocr_page(img) for img in images]
    pages_text = await asyncio.gather(*tasks)

    full_text = " ".join(pages_text)
    artifacts.cache_text(full_text)

    # fast-langdetect only needs first 2000 chars
    results = detect(full_text[:2000], model="lite")
    lang_result = results[0] if results else {"lang": "unknown", "score": 0}
    return full_text, lang_result