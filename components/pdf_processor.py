"""
pdf_processor.py — Parse PDF and store chunks in the vector store.

Changes from old version:
- langchain.text_splitter → langchain_text_splitters (new package)
- No other logic changes
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from components.vector_store import VectorStoreManager


class DocumentChunk:
    """Represents a single text chunk extracted from a PDF page."""

    def __init__(self, text: str, page_number: int, chunk_id: str):
        self.text = text
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.metadata = {
            "page_number": page_number,
            "chunk_id":    chunk_id,
        }


def process_pdf(uploaded_file) -> object:
    """
    Parse an uploaded PDF, split into chunks, embed and store them.

    Returns:
        The ChromaDB collection (vectorstore) object, ready for retrieval.
    """
    pdf_reader = PdfReader(uploaded_file)
    documents = []

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({"text": text, "page_number": page_num})

    # Chunk the raw page text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks: list[DocumentChunk] = []
    for doc in documents:
        raw_chunks = text_splitter.split_text(doc["text"])
        for chunk_text in raw_chunks:
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    page_number=doc["page_number"],
                    chunk_id=str(uuid.uuid4()),
                )
            )

    # Store in the singleton vector store
    vector_store = VectorStoreManager()
    return vector_store.add_pdf_chunks(chunks, uploaded_file.name)