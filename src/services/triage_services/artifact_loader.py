from pdf2image import convert_from_path
import pdfplumber
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

class DocumentArtifacts:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self._images = None
        self._pdf = None
        self._cached_text = None

    async def load_images(self):
        if self._images is None:
            loop = asyncio.get_running_loop()
            pages = await loop.run_in_executor(
                executor, lambda: convert_from_path(str(self.pdf_path), dpi=300)
            )
            self._images = [np.array(p) for p in pages]
        return self._images

    async def load_pdf(self):
        if self._pdf is None:
            loop = asyncio.get_running_loop()
            self._pdf = await loop.run_in_executor(executor, lambda: pdfplumber.open(str(self.pdf_path)))
        return self._pdf

    def cache_text(self, text: str):
        self._cached_text = text

    @property
    def cached_text(self):
        return self._cached_text