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
async def ingest_documents(
    input_path: str,
    recreate: bool = False,
    request_id: str | None = None,
    source_name: str | None = None,
    source_type: str | None = None,
):
    total_start = time.perf_counter()
    timings = {}
    trace = {
        "request_id": request_id,
        "operation": "ingest",
        "source": {
            "name": source_name or input_path,
            "type": source_type or "path",
        },
        "recreate": recreate,
        "providers": {
            "llm": app_settings.LLM_PROVIDER,
            "embedding": app_settings.DENSE_PROVIDER,
            "sparse": app_settings.SPARSE_PROVIDER,
            "reranker": app_settings.RERANKER_PROVIDER,
        },
        "models": {
            "llm": app_settings.LLM_MODEL,
            "embedding": app_settings.EMBEDDING_MODEL,
            "sparse": app_settings.SPARSE_MODEL,
            "reranker": app_settings.RERANKER_MODEL,
        },
        "retrieval_mode": app_settings.RETRIEVAL_MODE,
        "warnings": [
            "Reset/re-ingest when embedding provider, embedding model, or embedding dimension changes.",
        ],
    }

    try:
        logger.info(
            "ingestion_started",
            request_id=request_id,
            source_name=source_name or input_path,
            source_type=source_type or "path",
            recreate=recreate,
            llm_provider=app_settings.LLM_PROVIDER,
            embedding_provider=app_settings.DENSE_PROVIDER,
            retrieval_mode=app_settings.RETRIEVAL_MODE,
        )
        indexer = HybridIndexer()

        if recreate:
            start = time.perf_counter()
            deleted = await indexer.store_provider.delete_collection()
            timings["delete_collection"] = round(time.perf_counter() - start, 4)
            logger.info(
                "collection_delete_finished",
                request_id=request_id,
                collection_name=deleted["collection_name"],
                deleted=deleted["deleted"],
                existed=deleted.get("existed"),
                duration_seconds=timings["delete_collection"],
            )

        start = time.perf_counter()
        await indexer.store_provider.init_collection_if_needed()
        timings["init_collection"] = round(time.perf_counter() - start, 4)
        
        # ---- Load Documents
        start = time.perf_counter()
        documents = load_documents(input_path)
        timings["load_documents"] = round(time.perf_counter() - start, 4)
        logger.info(
            "documents_loaded",
            request_id=request_id,
            document_count=len(documents),
            duration_seconds=timings["load_documents"],
        )

        # ---- Chunking
        splitter = SentenceSplitter(
            chunk_size=app_settings.CHUNK_SIZE,
            chunk_overlap=app_settings.CHUNK_OVERLAP,
        )

        start = time.perf_counter()
        nodes = splitter.get_nodes_from_documents(documents)
        timings["chunking"] = round(time.perf_counter() - start, 4)
        logger.info(
            "nodes_created",
            request_id=request_id,
            chunk_count=len(nodes),
            duration_seconds=timings["chunking"],
        )

        # ---- Indexing (Dense + Sparse + Insert)
        start = time.perf_counter()
        index = indexer.build_index(nodes)
        timings["indexing"] = round(time.perf_counter() - start, 4)
        timings["total"] = round(time.perf_counter() - total_start, 4)
        logger.info(
            "index_built",
            request_id=request_id,
            duration_seconds=timings["indexing"],
            total_duration_seconds=timings["total"],
        )

        return {
            "status": "success",
            "docs_ingested": len(documents),
            "nodes": len(nodes),
            "trace": {
                **trace,
                "status": "success",
                "document_count": len(documents),
                "chunk_count": len(nodes),
                "timings": timings,
            },
        }

    except Exception as e:
        timings["total"] = round(time.perf_counter() - total_start, 4)
        logger.error(
            "ingestion_failed",
            request_id=request_id,
            error=str(e),
            duration_seconds=timings["total"],
            exc_info=True,
        )
        raise
