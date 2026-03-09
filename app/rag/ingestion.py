from pathlib import Path
import re
import time
from app.rag.hybrid_indexer import HybridIndexer
import structlog
from app.config import app_settings, configure_llm_settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, PyMuPDFReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

from rag.vectorstores.factory import get_vector_store_provider

logger = structlog.get_logger()

configure_llm_settings()

# ------------------------
# Text Cleaning
# ------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"endobj.*?obj", "", text, flags=re.DOTALL)
    text = re.sub(r"/Type\s*/\w+", "", text)
    return text.strip()

# ------------------------
# Dynamic file Loader (Production Robust)
# ------------------------

def load_documents(input_path: str):
    input_path = Path(input_path)
    documents = []

    pdf_files = list(input_path.glob("**/*.pdf"))
    other_files = list(input_path.glob("**/*.*"))

    # ---- PDFs
    for file in pdf_files:
        try:
            reader = PyMuPDFReader()
            docs = reader.load_data(file_path=str(file))
            logger.info("Loaded PDF with PyMuPDF", file=str(file))
        except Exception:
            reader = PDFReader()
            docs = reader.load_data(file=str(file))
            logger.warning("Fallback to PDFReader", file=str(file))

        cleaned_docs = [
            Document(
                text=clean_text(d.text),
                metadata=d.metadata,
            )
            for d in docs
        ]

        documents.extend(cleaned_docs)

    # ---- MD / TXT
    non_pdf_files = [
        f for f in other_files
        if f.suffix.lower() in [".md", ".txt"]
    ]

    if non_pdf_files:
        reader = SimpleDirectoryReader(
            input_files=[str(f) for f in non_pdf_files]
        )
        docs = reader.load_data()

        cleaned_docs = [
            Document(
                text=clean_text(d.text),
                metadata=d.metadata,
            )
            for d in docs
        ]

        documents.extend(cleaned_docs)

    if not documents:
        supported = [".pdf", ".txt", ".md", ".docx", ".doc", ".html"]  # adjust to your loader
        raise ValueError(f"No supported documents found in '{input_path}'. Supported: {supported}")

    return documents

# ------------------------
# Ingest Pipeline
# ------------------------
async def ingest_documents(input_path: str, recreate: bool = False):
    try:
        logger.info("Ingesting file.", file_path=input_path)
        indexer = HybridIndexer()

        if recreate:
            deleted = await indexer.store_provider.delete_collection()
            logger.info(f"{deleted["collection_name"]} successfully deleted.") if deleted["deleted"] else logger.info(f"Failed to delete {deleted["collection_name"]}.")

        await indexer.store_provider.init_collection_if_needed()
        
        # ---- Load Documents
        start = time.time()
        documents = load_documents(input_path)
        logger.info("Documents loaded", count=len(documents), seconds=time.time() - start)

        # ---- Chunking
        splitter = SentenceSplitter(
            chunk_size=app_settings.CHUNK_SIZE,
            chunk_overlap=app_settings.CHUNK_OVERLAP,
        )

        start = time.time()
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info("Nodes created", count=len(nodes), seconds=time.time() - start)

        # ---- Indexing (Dense + Sparse + Insert)
        start = time.time()
        index = indexer.build_index(nodes)
        logger.info("Index built", seconds=time.time() - start)

        return {
            "status": "success",
            "docs_ingested": len(documents),
            "nodes": len(nodes),
        }

    except Exception as e:
        logger.error("Ingest failed", error=str(e), exc_info=True)
        raise
