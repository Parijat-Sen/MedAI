"""
============================================================
MedAI — RAG Pipeline (FAISS + sentence-transformers)
============================================================
Implements Retrieval-Augmented Generation for medical
knowledge lookup.

Pipeline:
  1. Load medical documents
  2. Chunk into passages
  3. Embed with sentence-transformers
  4. Store in FAISS vector index
  5. Retrieve top-K relevant passages on query
============================================================
"""

import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
DATA_DIR = Path("backend/data")
INDEX_DIR = Path("backend/rag/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 80    # overlap between chunks


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""
    content: str
    source: str
    score: float
    chunk_id: int


# ══════════════════════════════════════════════════════════
# DOCUMENT CHUNKER
# ══════════════════════════════════════════════════════════

class MedicalDocumentChunker:
    """
    Splits medical documents into overlapping chunks for embedding.
    Sentence-aware splitting preserves medical context.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split a document into overlapping chunks."""
        # Split on paragraph breaks first for better coherence
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append({"content": current_chunk, "source": source})
                # Start new chunk with overlap
                if len(para) > self.chunk_size:
                    # Para itself is too large — split by sentence
                    sentences = para.replace(". ", ".|").split("|")
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) <= self.chunk_size:
                            sub_chunk += sent + " "
                        else:
                            if sub_chunk:
                                chunks.append({"content": sub_chunk.strip(), "source": source})
                            sub_chunk = sent + " "
                    current_chunk = sub_chunk.strip()
                else:
                    # Use overlap from previous chunk
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + para

        if current_chunk.strip():
            chunks.append({"content": current_chunk.strip(), "source": source})

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["content"], doc.get("title", "Unknown"))
            all_chunks.extend(chunks)
        logger.info(f"Chunked {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks


# ══════════════════════════════════════════════════════════
# FAISS VECTOR STORE
# ══════════════════════════════════════════════════════════

class MedicalFAISSStore:
    """
    FAISS-backed vector store for medical document retrieval.
    Uses cosine similarity (inner product on normalized vectors).
    """

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.encoder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[Dict] = []  # Parallel array of text chunks
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def build_index(self, chunks: List[Dict]):
        """
        Build FAISS index from document chunks.

        Args:
            chunks: List of {"content": str, "source": str}
        """
        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        self.chunks = chunks

        # Embed all chunks (batched for efficiency)
        texts = [c["content"] for c in chunks]
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity via inner product
        )

        # Create FAISS index (IndexFlatIP = inner product on normalized = cosine)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))

        logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors")

    def save(self, path: Path = INDEX_DIR):
        """Persist index and chunks to disk."""
        faiss.write_index(self.index, str(path / "medical.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"💾 FAISS index saved to {path}")

    def load(self, path: Path = INDEX_DIR):
        """Load index and chunks from disk."""
        index_path = path / "medical.index"
        chunks_path = path / "chunks.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build_rag_index() first.")

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"✅ FAISS index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve top-K most relevant chunks for a query.

        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve

        Returns:
            List of RetrievedChunk ordered by relevance (descending)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load() or build_index() first.")

        # Embed query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                content=chunk["content"],
                source=chunk["source"],
                score=float(score),
                chunk_id=int(idx)
            ))

        return results


# ══════════════════════════════════════════════════════════
# RAG PIPELINE (main interface)
# ══════════════════════════════════════════════════════════

class RAGPipeline:
    """
    High-level RAG interface used by the FastAPI backend.
    Manages the full retrieve → format → return cycle.
    """

    def __init__(self):
        self.store = MedicalFAISSStore()
        self._loaded = False

    def initialize(self):
        """Load existing FAISS index or build from documents."""
        index_path = INDEX_DIR / "medical.index"

        if index_path.exists():
            logger.info("Loading existing FAISS index...")
            self.store.load()
        else:
            logger.info("FAISS index not found. Building from documents...")
            self._build_from_documents()

        self._loaded = True

    def _build_from_documents(self):
        """Build FAISS index from medical documents."""
        doc_path = DATA_DIR / "medical_documents.json"
        if not doc_path.exists():
            raise FileNotFoundError(
                "medical_documents.json not found. "
                "Run: python backend/data/generate_dataset.py first."
            )

        with open(doc_path) as f:
            documents = json.load(f)

        chunker = MedicalDocumentChunker()
        chunks = chunker.chunk_documents(documents)
        self.store.build_index(chunks)
        self.store.save()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve relevant medical knowledge for a query."""
        if not self._loaded:
            self.initialize()
        return self.store.retrieve(query, top_k=top_k)

    def retrieve_for_symptoms(self, symptoms: List[str], diseases: List[str],
                               top_k: int = 5) -> str:
        """
        Retrieve medical context relevant to given symptoms and predicted diseases.
        Returns formatted context string for LLM injection.

        Args:
            symptoms: List of symptom strings
            diseases: List of predicted disease names
            top_k: Chunks to retrieve

        Returns:
            Formatted context string
        """
        # Build a rich query combining symptoms and diseases
        symptom_str = ", ".join(symptoms[:8])  # Limit for clarity
        disease_str = ", ".join(diseases[:3])
        query = f"Symptoms: {symptom_str}. Possible diseases: {disease_str}. Treatment and diagnosis."

        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return "No relevant medical context found."

        # Format for LLM consumption
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.source} | Relevance: {chunk.score:.2f}]\n{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def rebuild_index(self):
        """Force rebuild of FAISS index (e.g., after adding new documents)."""
        logger.info("Rebuilding FAISS index...")
        # Remove existing index
        for f in INDEX_DIR.iterdir():
            f.unlink()
        self._build_from_documents()


# ══════════════════════════════════════════════════════════
# BUILD SCRIPT
# ══════════════════════════════════════════════════════════

def build_rag_index():
    """Standalone function to build the RAG index from scratch."""
    logger.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
    logger.info("\n" + "="*60)
    logger.info("   MedAI — Building RAG Index")
    logger.info("="*60)

    pipeline = RAGPipeline()
    pipeline._build_from_documents()

    # Quick test
    logger.info("\nRunning retrieval test...")
    test_query = "fever headache malaria treatment"
    results = pipeline.retrieve(test_query, top_k=3)

    logger.info(f"\nQuery: '{test_query}'")
    for r in results:
        logger.info(f"  [{r.score:.3f}] {r.source}: {r.content[:100]}...")

    logger.info("\n✅ RAG index built and tested successfully!")


if __name__ == "__main__":
    build_rag_index()