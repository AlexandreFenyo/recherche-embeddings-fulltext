from qdrant_client import QdrantClient
from opensearchpy import OpenSearch, RequestsHttpConnection
from embedder import ModelManager

QDRANT_PATH = "./qdrant_storage"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OS_INDEX = "documents"


class Searcher:
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

    def search_embeddings(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        n_candidates: int = 20,
        hyde: bool = False,
        hyde_text: str = "",
    ) -> list[dict]:
        """Recherche dans TOUTES les collections, en tagguant chaque résultat
        par sa collection d'origine (modèle × stratégie).

        `top_k` est le nombre de résultats *par collection*. Les lignes sont
        regroupées par collection (ordre de balayage).

        Si `rerank` est vrai, on récupère `n_candidates` chunks par collection,
        on les reclasse avec le cross-encoder (BAAI/bge-reranker-v2-m3) — le
        reranking évalue toujours la **requête originale** — et on garde les
        `top_k` meilleurs.

        Si `hyde` est vrai, chaque collection est interrogée DEUX fois : avec la
        requête brute (variante « requête seule ») et avec le pseudo-document
        `hyde_text` (variante « HyDE »), pour comparer ce que HyDE apporte.
        """
        collections = sorted(c.name for c in self.qdrant.get_collections().collections)
        if not collections:
            return []

        reranker = self.embedder.get_reranker() if rerank else None

        # Variantes de texte à encoder comme « requête ».
        variants = [("requête seule", query)]
        if hyde and hyde_text:
            variants.append(("HyDE", hyde_text))

        vec_cache: dict[tuple, list[float]] = {}
        rows: list[dict] = []
        for col in collections:
            # Le modèle est encodé en préfixe du nom ("{modèle}__{stratégie}").
            model_name = col.split("__", 1)[0]
            if model_name not in self.embedder.models:
                continue

            for vlabel, vtext in variants:
                key = (model_name, vlabel)
                if key not in vec_cache:
                    vec_cache[key] = self.embedder.embed_texts(
                        model_name, [vtext], mode="query"
                    )[0].tolist()

                # Sans rerank : top_k direct. Avec rerank : un vivier à reclasser.
                limit = max(n_candidates, top_k) if rerank else top_k
                hits = self.qdrant.query_points(
                    collection_name=col,
                    query=vec_cache[key],
                    limit=limit,
                    with_payload=True,
                ).points

                ce_scores: list[float | None] = [None] * len(hits)
                if rerank and hits:
                    pairs = [(query, h.payload.get("text", "")) for h in hits]
                    scores = reranker.predict(pairs)
                    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:top_k]
                    hits = [hits[i] for i in order]
                    ce_scores = [round(float(scores[i]), 4) for i in order]

                for h, ce in zip(hits, ce_scores):
                    rows.append(
                        {
                            "Modèle": h.payload.get("model", model_name),
                            "Stratégie": h.payload.get("strategy", ""),
                            "Variante": vlabel,
                            "Score": round(h.score, 4),
                            "Score CE": ce if ce is not None else "",
                            "Fichier": h.payload.get("filename", ""),
                            "Chunk": h.payload.get("chunk_index", ""),
                            "Texte": h.payload.get("text", "")[:400],
                        }
                    )
        return rows

    def search_fulltext(self, query: str, top_k: int = 10) -> list[dict]:
        client = self._get_opensearch()
        try:
            resp = client.search(
                index=OS_INDEX,
                body={
                    "size": top_k,
                    "query": {
                        "query_string": {
                            "query": query,
                            "default_field": "content",
                            "analyze_wildcard": True,
                        }
                    },
                    "highlight": {
                        # Pour les très gros documents, OpenSearch refuse d'analyser
                        # au-delà de index.highlight.max_analyzed_offset (1 000 000
                        # par défaut) et lève une erreur. On borne explicitement
                        # l'analyse au début du texte : pas d'erreur, surlignage
                        # produit sur la portion analysée.
                        # NB : dans cette version d'OpenSearch le paramètre s'écrit
                        # « max_analyzer_offset » (et non « max_analyzed_offset »).
                        "max_analyzer_offset": 1000000,
                        "fields": {"content": {"fragment_size": 300, "number_of_fragments": 2}},
                    },
                },
            )
        except Exception as e:
            return [{"Erreur": str(e)}]

        results = []
        for hit in resp["hits"]["hits"]:
            highlights = hit.get("highlight", {}).get("content", [])
            results.append({
                "Score": round(hit["_score"], 4),
                "Fichier": hit["_source"].get("filename", ""),
                "Chunk": hit["_source"].get("chunk_index", ""),
                "Extraits": " … ".join(highlights) if highlights else hit["_source"].get("content", "")[:300],
            })
        return results

    def list_indexed_strategies(self) -> list[dict]:
        rows = []
        for col in self.qdrant.get_collections().collections:
            info = self.qdrant.get_collection(col.name)
            rows.append({
                "Collection": col.name,
                "Vecteurs": info.points_count,
            })
        return rows
