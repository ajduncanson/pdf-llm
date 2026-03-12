from typing import List, Tuple


class VectorStore:
    """
    Ephemeral ChromaDB collection for a single query session.
    Created fresh each run — no persistence between CLI invocations.
    Uses cosine distance so similarity scores are in [0, 1].
    """

    def __init__(self):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb package required: pip install chromadb")

        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))],
        )

    def query(
        self, embedding: List[float], top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Returns (documents, similarities) where similarity = 1 - cosine_distance,
        so 1.0 = identical, 0.0 = orthogonal.
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
        )
        docs = results["documents"][0]
        # ChromaDB cosine distance is in [0, 1]; convert to similarity
        similarities = [1.0 - d for d in results["distances"][0]]
        return docs, similarities
