"""ChromaDB RAG Tool — Google Gemini Embeddings (production).

Uses GoogleGenerativeAiEmbeddingFunction (via chromadb.utils.embedding_functions)
with the GOOGLE_API_KEY already configured in the environment.

No offline / deterministic fallback embedding functions are included.
If GOOGLE_API_KEY is missing, initialization raises a clear error so the
operator can fix the configuration.
"""

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb import EmbeddingFunction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from yatra_agent.config.settings import (
    CHROMA_PERSIST_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
    TRAVEL_DOCS_DIR,
)

logger = logging.getLogger("yatra_ai.rag")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Google Gemini embedding function using official google.genai SDK."""

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Any) -> Any:
        if not input:
            return []
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=list(input),
        )
        return [e.values for e in resp.embeddings]

    def embed_query(self, input: Any) -> Any:
        return self(input)

    def embed_documents(self, input: Any) -> Any:
        return self(input)


def _get_embedding_function() -> GeminiEmbeddingFunction:
    """Return a Google Gemini embedding function.

    Requires GOOGLE_API_KEY to be set in the environment.
    Raises EnvironmentError if the key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Set it in your .env file to enable the RAG knowledge base."
        )
    return GeminiEmbeddingFunction(
        api_key=api_key,
        model_name="models/gemini-embedding-001",
    )


# ---------------------------------------------------------------------------
# TravelRAGTool Class
# ---------------------------------------------------------------------------
class TravelRAGTool:
    """Manages ChromaDB vector database for Indian travel knowledge retrieval."""

    COLLECTION_NAME = "yatra_travel_guides"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedding_fn = _get_embedding_function()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def load_file_content(self, file_path: Path) -> str:
        """Extract plain text from supported document types (.md, .txt, .pdf, .csv)."""
        suffix = file_path.suffix.lower()

        if suffix in (".md", ".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text_parts = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"--- Page {idx + 1} ---\n{page_text}")
                return "\n\n".join(text_parts)
            except Exception as e:
                logger.error(f"Failed to read PDF {file_path}: {e}")
                return ""

        elif suffix == ".csv":
            try:
                rows = []
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    for row in reader:
                        if headers:
                            formatted = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
                            rows.append(formatted)
                        else:
                            rows.append(", ".join(row))
                return "\n".join(rows)
            except Exception as e:
                logger.error(f"Failed to read CSV {file_path}: {e}")
                return ""

        return ""

    def add_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Split and add arbitrary text into the knowledge base with deduplication."""
        if not text.strip():
            return 0

        chunks = self.splitter.split_text(text)
        if not chunks:
            return 0

        base_meta = metadata.copy() if metadata else {}
        base_meta["source"] = source

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}:{idx}:{chunk[:100]}".encode("utf-8")).hexdigest()
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            meta["char_length"] = len(chunk)

            documents.append(chunk)
            metadatas.append(meta)
            ids.append(chunk_id)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def build_knowledge_base(
        self,
        data_dir: Optional[str] = None,
        force_recreate: bool = False,
    ) -> Dict[str, Any]:
        """Ingest all documents from the travel_docs directory into ChromaDB."""
        docs_path = Path(data_dir or TRAVEL_DOCS_DIR)
        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)
            return {"documents_indexed": 0, "sources": [], "chunks_added": 0}

        if force_recreate:
            try:
                self.client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        supported_exts = {".md", ".txt", ".pdf", ".csv"}
        total_chunks = 0
        indexed_sources = []

        for file_path in docs_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                content = self.load_file_content(file_path)
                if content.strip():
                    fname = file_path.stem.lower()
                    dest_guess = fname.replace("_guide", "").replace("_travel", "").replace("_", " ").title()
                    meta = {
                        "filename": file_path.name,
                        "destination_tag": dest_guess,
                    }
                    added = self.add_text(content, source=file_path.name, metadata=meta)
                    total_chunks += added
                    indexed_sources.append(file_path.name)
                    logger.info(f"Indexed {file_path.name}: {added} chunks")

        return {
            "documents_indexed": len(indexed_sources),
            "sources": indexed_sources,
            "chunks_added": total_chunks,
        }

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search vector database for relevant travel guide chunks."""
        if not query.strip() or self.collection.count() == 0:
            return []

        try:
            query_kwargs: Dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k, self.collection.count()),
            }
            if filter_metadata:
                query_kwargs["where"] = filter_metadata

            results = self.collection.query(**query_kwargs)

            docs: List[Document] = []
            if results and results.get("documents") and results["documents"][0]:
                raw_docs = results["documents"][0]
                raw_metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(raw_docs)

                for text, meta in zip(raw_docs, raw_metas):
                    docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception as e:
            logger.error(f"RAG search error for '{query}': {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return knowledge base status, chunk count, and unique sources."""
        count = self.collection.count()
        sources: set = set()
        if count > 0:
            try:
                data = self.collection.get(limit=min(count, 500), include=["metadatas"])
                for m in data.get("metadatas", []):
                    if m and "source" in m:
                        sources.add(m["source"])
            except Exception:
                pass

        return {
            "total_chunks": count,
            "unique_sources_count": len(sources),
            "sources": sorted(list(sources)),
        }

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a particular document source."""
        try:
            self.collection.delete(where={"source": source})
            return 1
        except Exception as e:
            logger.error(f"Error deleting source '{source}': {e}")
            return 0
