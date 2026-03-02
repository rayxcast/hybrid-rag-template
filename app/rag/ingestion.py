from pathlib import Path
from typing import List, Tuple
import re
import time
import structlog
from app.config import app_settings, configure_llm_settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PDFReader, PyMuPDFReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from qdrant_client import QdrantClient, models
from app.rag.embedding_providers.sparse.factory import get_sparse_provider

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
# Sparse Embeddings
# ------------------------
class HybridIndexer:
    def __init__(self):
        self.sparse = get_sparse_provider(app_settings.SPARSE_PROVIDER)

    def custom_sparse_embed(self, texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        return self.sparse.embed_documents(texts)

    def custom_sparse_query(self, texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        return self.sparse.embed_query(texts)


# ------------------------
# Vector Store
# ------------------------

def get_vector_store():
    client = QdrantClient(url=app_settings.QDRANT_URL)
    indexer = HybridIndexer()

    return QdrantVectorStore(
        client=client,
        collection_name=app_settings.COLLECTION_NAME,
        enable_hybrid=True,
        sparse_doc_fn=indexer.custom_sparse_embed,
        sparse_query_fn=indexer.custom_sparse_query,
        text_sparse_name="text-sparse",
        use_default_sparse_query_encoder=False,
    )


def init_collection_if_needed():
    client = QdrantClient(url=app_settings.QDRANT_URL)

    if not client.collection_exists(app_settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=app_settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=app_settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams()
            },
        )
        logger.info("created_qdrant_collection", name=app_settings.COLLECTION_NAME)


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
        if recreate:
            client = QdrantClient(url=app_settings.QDRANT_URL)
            client.delete_collection(app_settings.COLLECTION_NAME)

        init_collection_if_needed()
        vector_store = get_vector_store()

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

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
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.info("Index built", seconds=time.time() - start)

        return {
            "status": "success",
            "docs_ingested": len(documents),
            "nodes": len(nodes),
        }

    except Exception as e:
        logger.error("Ingest failed", error=str(e), exc_info=True)
        raise
