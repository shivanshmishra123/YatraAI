"""ChromaDB RAG Tool for Indian travel knowledge retrieval."""

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from yatra_agent.config.settings import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
    TRAVEL_DOCS_DIR,
)

logger = logging.getLogger("yatra_ai.rag")


from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


STOPWORDS = {
    "what", "are", "the", "in", "for", "is", "a", "an", "and", "or", "to", "of",
    "on", "with", "at", "by", "from", "how", "tell", "me", "about", "can", "i",
    "you", "do", "does", "did", "this", "that", "these", "those", "there", "where",
}


class DeterministicLocalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fast, deterministic word-hash embedding function for offline/test environments."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        import re
        embeddings = []
        for text in input:
            vec = [0.0] * self.dimensions
            clean_text = re.sub(r"[^\w\s]", " ", text.lower())
            words = [w for w in clean_text.split() if w not in STOPWORDS]
            if not words:
                words = clean_text.split()
            if not words:
                embeddings.append(vec)
                continue
            for word in words:
                idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dimensions
                vec[idx] += 1.0

            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

    @staticmethod
    def name() -> str:
        return "deterministic_local"

    def get_config(self) -> Dict[str, Any]:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "DeterministicLocalEmbeddingFunction":
        return DeterministicLocalEmbeddingFunction(dimensions=config.get("dimensions", EMBEDDING_DIMENSIONS))


def _get_embedding_function():
    """Return OpenAIEmbeddingFunction if API key is valid, else deterministic local fallback."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not openai_key.startswith("sk-your") and len(openai_key) > 20:
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name=EMBEDDING_MODEL,
            )
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI embedding function, using fallback: {e}")

    return DeterministicLocalEmbeddingFunction()


# ---------------------------------------------------------------------------
# TravelRAGTool Class
# ---------------------------------------------------------------------------
class TravelRAGTool:
    """Manages ChromaDB vector database for Indian travel knowledge retrieval."""

    COLLECTION_NAME = "yatra_travel_guides"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedding_fn = _get_embedding_function()
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
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
            # MD5 hash of source:idx:content[:100] for idempotent deduplication
            chunk_id = hashlib.md5(f"{source}:{idx}:{chunk[:100]}".encode("utf-8")).hexdigest()
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            meta["char_length"] = len(chunk)

            documents.append(chunk)
            metadatas.append(meta)
            ids.append(chunk_id)

        # Upsert into ChromaDB
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
        """Ingest all documents from travel_docs directory into ChromaDB."""
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
                    # Infer destination or topic from filename
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
            logger.error(f"Error during RAG search for '{query}': {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return knowledge base status, chunk count, and unique sources."""
        count = self.collection.count()
        sources: set = set()
        if count > 0:
            try:
                # Sample up to 500 metadatas to find unique sources
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
