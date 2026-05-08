import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class EmbeddingStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="causality_nodes",
            embedding_function=self._ef,
        )

    def upsert_node(self, node_id: str, text: str, metadata: dict) -> None:
        self._collection.upsert(
            ids=[node_id],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        k = min(top_k, count)
        results = self._collection.query(query_texts=[query], n_results=k)
        output = []
        for i, node_id in enumerate(results["ids"][0]):
            output.append({
                "id": node_id,
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return output
