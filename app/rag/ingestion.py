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
from fastembed import SparseTextEmbedding

logger = structlog.get_logger()
configure_llm_settings()

# ------------------------
# Embeddings (Production Mode)
# ------------------------

sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

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

def custom_sparse_embed(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
    embeddings = list(sparse_model.embed(texts))
    indices_list = []
    values_list = []

    for emb in embeddings:
        indices_list.append(emb.indices.tolist())
        values_list.append(emb.values.tolist())

    return indices_list, values_list


def custom_sparse_query(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
    emb = next(sparse_model.embed(texts))
    return [emb.indices.tolist()], [emb.values.tolist()]


# ------------------------
# Vector Store
# ------------------------

def get_vector_store():
    client = QdrantClient(url=app_settings.QDRANT_URL)

    return QdrantVectorStore(
        client=client,
        collection_name=app_settings.COLLECTION_NAME,
        enable_hybrid=True,
        sparse_doc_fn=custom_sparse_embed,
        sparse_query_fn=custom_sparse_query,
        text_sparse_name="text-sparse",
        use_default_sparse_query_encoder=False,
    )


def init_collection_if_needed():
    client = QdrantClient(url=app_settings.QDRANT_URL)

    if not client.collection_exists(app_settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=app_settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,
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
            chunk_size=1024,
            chunk_overlap=150,
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
