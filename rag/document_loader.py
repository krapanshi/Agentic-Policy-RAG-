"""
document_loader.py
------------------
Reads documents (TXT, PDF, Markdown) from a folder and splits them into
smaller overlapping "chunks" that can be stored in the vector database.

"""

import os
import logging
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """
    Loads documents from a folder and returns them as a list of chunk dicts.

    Each chunk dict looks like:
        {
            "content":  "...the actual text...",
            "source":   "leave_policy.txt",
            "chunk_id": 3,
            "page":     ""   # page number for PDFs, empty string for others
        }
    """

    def __init__(
        self,
        docs_folder: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.docs_folder = docs_folder
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    def load_all(self) -> List[Dict]:
        """
        Walk the docs_folder and load every supported file.
        Returns a flat list of chunk dicts ready to be embedded.
        """
        folder = Path(self.docs_folder)
        if not folder.exists():
            logging.warning(f"Documents folder not found: {self.docs_folder}")
            return []

        all_chunks: List[Dict] = []
        supported = {".txt", ".md", ".pdf"}

        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() not in supported:
                continue

            if file_path.suffix.lower() == ".pdf":
                chunks = self._load_pdf(file_path)
            else:
                chunks = self._load_text(file_path)

            all_chunks.extend(chunks)
            logging.info(f"Loaded {len(chunks)} chunks from '{file_path.name}'")

        logging.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _load_text(self, file_path: Path) -> List[Dict]:
        """Load a plain-text or Markdown file and chunk it."""
        try:
            text = file_path.read_text(encoding="utf-8")
            return self._make_chunks(text, file_path.name, page=None)
        except Exception as exc:
            logging.error(f"Error loading '{file_path.name}': {exc}")
            return []

    def _load_pdf(self, file_path: Path) -> List[Dict]:
        """Load a PDF file page-by-page and chunk each page."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            chunks: List[Dict] = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    page_chunks = self._make_chunks(
                        text, file_path.name, page=page_num
                    )
                    chunks.extend(page_chunks)

            return chunks
        except Exception as exc:
            logging.error(f"Error loading PDF '{file_path.name}': {exc}")
            return []

    def _make_chunks(
        self, text: str, source_name: str, page
    ) -> List[Dict]:
        """
        Split raw text into chunks using LangChain's RecursiveCharacterTextSplitter.
        Each chunk gets metadata: source filename, chunk_id, and optional page number.
        """
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                chunks.append(
                    {
                        "content": chunk_text.strip(),
                        "source": source_name,
                        "chunk_id": idx,
                        "page": page if page is not None else "",
                    }
                )
        return chunks
