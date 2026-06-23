import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from embedder import ModelManager, MODELS_CONFIG
from naming import collection_name

QDRANT_PATH = "./qdrant_storage"  # ancien mode embarqué (conservé pour référence)
QDRANT_URL = "http://localhost:6333"  # mode serveur (console sur :6333/dashboard)
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OS_INDEX = "documents"


class IndexManager:
    def __init__(self, embedder: ModelManager, qdrant: QdrantClient):
        self.embedder = embedder
        self.qdrant = qdrant
        self._os: OpenSearch | None = None

    def _get_opensearch(self) -> OpenSearch:
        if self._os is None:
            self._os = OpenSearch(
                hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
                connection_class=RequestsHttpConnection,
                use_ssl=False,
                verify_certs=False,
            )
        return self._os

    def opensearch_available(self) -> bool:
        try:
            client = self._get_opensearch()
            return client.ping()
        except Exception:
            return False

    def _ensure_collection(self, model_name: str, strategy_name: str):
        name = collection_name(model_name, strategy_name)
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if name not in existing:
            dims = MODELS_CONFIG[model_name]["dims"]
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )
        return name

    def _ensure_os_index(self):
        client = self._get_opensearch()
        if not client.indices.exists(index=OS_INDEX):
            client.indices.create(
                index=OS_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "filename": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "strategy": {"type": "keyword"},
                            "content": {"type": "text", "analyzer": "standard"},
                        }
                    }
                },
            )

    def clear_all(self):
        for col in self.qdrant.get_collections().collections:
            self.qdrant.delete_collection(col.name)

        if self.opensearch_available():
            client = self._get_opensearch()
            if client.indices.exists(index=OS_INDEX):
                client.indices.delete(index=OS_INDEX)

    def index_document(
        self,
        content: str,
        filename: str,
        chunker,
        strategy_name: str,
        progress_cb=None,
    ) -> dict:
        chunks = chunker.chunk(content)
        if not chunks:
            return {"error": "Aucun chunk produit"}

        total_points = 0
        for model_name in self.embedder.models:
            if progress_cb:
                progress_cb(f"Embeddings {model_name} ({len(chunks)} chunks)…")

            col = self._ensure_collection(model_name, strategy_name)

            # Purge les points existants de ce fichier pour éviter l'accumulation
            # de doublons à chaque ré-indexation.
            self.qdrant.delete(
                collection_name=col,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                    )
                ),
            )

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_texts(model_name, texts, mode="passage")

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": chunks[i]["text"],
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "strategy": strategy_name,
                        "model": model_name,
                        **chunks[i].get("metadata", {}),
                    },
                )
                for i in range(len(chunks))
            ]
            self.qdrant.upsert(collection_name=col, points=points)
            total_points += len(points)

        os_count = 0
        if self.opensearch_available():
            if progress_cb:
                progress_cb(f"Indexation OpenSearch ({len(chunks)} chunks)…")
            self._ensure_os_index()
            client = self._get_opensearch()
            # Purge tous les chunks existants de ce fichier (quelle que soit la
            # stratégie précédente) pour qu'une ré-indexation remplace proprement.
            client.delete_by_query(
                index=OS_INDEX,
                body={"query": {"term": {"filename": filename}}},
                ignore=[404],
            )
            # Un document OpenSearch par chunk (même découpage que les embeddings),
            # sous un id déterministe pour rester idempotent.
            actions = [
                {
                    "_index": OS_INDEX,
                    "_id": f"{filename}#{i}",
                    "_source": {
                        "filename": filename,
                        "chunk_index": i,
                        "strategy": strategy_name,
                        "content": chunks[i]["text"],
                    },
                }
                for i in range(len(chunks))
            ]
            helpers.bulk(client, actions, refresh=True)
            os_count = len(actions)

        return {
            "chunks": len(chunks),
            "models": list(self.embedder.models.keys()),
            "qdrant_points": total_points,
            "opensearch_chunks": os_count,
        }
