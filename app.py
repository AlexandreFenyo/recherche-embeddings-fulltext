import gradio as gr
import pandas as pd
from pathlib import Path

from qdrant_client import QdrantClient
from embedder import ModelManager, MODELS_CONFIG
from chunkers import CHUNKERS, FixedSizeChunker
from indexer import IndexManager, QDRANT_URL
from searcher import Searcher

# La stratégie « taille fixe » est la première du dict : elle est la seule
# paramétrée par les sliders (taille de chunk / overlap).
FIXED_SIZE_LABEL = list(CHUNKERS.keys())[0]


def _effective_strategy(strategy_name: str, chunk_size, overlap):
    """Nom de stratégie effectif + paramètres normalisés.

    Pour la taille fixe, le nom inclut les valeurs réelles des sliders afin que
    chaque jeu de paramètres ait sa propre collection (sinon des découpages
    différents se mélangeaient dans une même collection au libellé trompeur).
    L'overlap est borné à chunk_size - 1 pour rester cohérent avec le chunker.
    """
    cs = max(1, int(chunk_size))
    ov = max(0, min(int(overlap), cs - 1))
    if strategy_name == FIXED_SIZE_LABEL:
        return f"Taille fixe ({cs} mots, overlap {ov})", cs, ov
    return strategy_name, cs, ov


embedder = ModelManager()
qdrant_client = QdrantClient(url=QDRANT_URL)
indexer: IndexManager | None = None
searcher: Searcher | None = None
_status_log = []


def _log(msg: str):
    _status_log.append(msg)
    if len(_status_log) > 100:
        _status_log.pop(0)


def _get_indexer() -> IndexManager:
    global indexer
    if indexer is None:
        indexer = IndexManager(embedder, qdrant_client)
    return indexer


def _get_searcher() -> Searcher:
    global searcher
    if searcher is None:
        searcher = Searcher(embedder, qdrant_client)
    return searcher


def _models_status_df() -> pd.DataFrame:
    return pd.DataFrame(embedder.status())


def _collections_df() -> pd.DataFrame:
    return pd.DataFrame(_get_searcher().list_indexed_strategies())


def _refresh_on_load():
    """État rafraîchi à chaque (re)chargement de page : modèles chargés + collections."""
    return _models_status_df(), _collections_df()


def load_models(progress=gr.Progress()):
    # Ne renvoie QUE le texte de statut : la table « Modèles d'embeddings » est
    # rafraîchie après coup (via .then) pour que l'avancement ne s'affiche que
    # dans « Statut », pas en surimpression sur la table.
    _status_log.clear()
    n = len(MODELS_CONFIG)
    counter = [0]

    def cb(msg):
        _log(msg)
        frac = counter[0] / n
        progress(frac, desc=msg)
        counter[0] = min(counter[0] + 1, n)

    embedder.load_all(progress_cb=cb)
    return "Modèles chargés. OpenSearch : " + (
        "✓ disponible" if _get_indexer().opensearch_available() else "✗ non disponible"
    )


def unload_models():
    embedder.unload_all()
    return (
        pd.DataFrame(embedder.status()),
        "Modèles déchargés. VRAM GPU libérée.",
    )


def clear_all():
    _get_indexer().clear_all()
    return "Tout effacé.", pd.DataFrame(_get_searcher().list_indexed_strategies())


def index_document(file, strategy_name: str, chunk_size: int, overlap: int):
    # Ne renvoie QUE le texte : la table « Collections Qdrant indexées » est
    # rafraîchie après coup (via .then) pour éviter d'afficher l'activité
    # d'indexation en surimpression sur la table.
    if file is None:
        return "Aucun fichier sélectionné."

    if not embedder.models:
        return "Chargez d'abord les modèles."

    content = Path(file.name).read_text(encoding="utf-8")
    filename = Path(file.name).name

    # Construit le chunker avec les paramètres de l'UI pour la taille fixe.
    effective_strategy, cs, ov = _effective_strategy(strategy_name, chunk_size, overlap)
    if strategy_name == FIXED_SIZE_LABEL:
        chunker = FixedSizeChunker(cs, ov)
    else:
        chunker = CHUNKERS[strategy_name](embedder)

    logs = []

    def cb(msg):
        logs.append(msg)

    result = _get_indexer().index_document(content, filename, chunker, effective_strategy, progress_cb=cb)

    if "error" in result:
        return result["error"]

    summary = (
        f"✓ {filename} — {result['chunks']} chunks, "
        f"{result['qdrant_points']} vecteurs Qdrant, "
        f"{result.get('opensearch_chunks', 0)} chunks OpenSearch\n"
        + "\n".join(logs)
    )
    return summary


def search_embeddings(query: str, top_k: int, rerank: bool, n_candidates: int, hyde: bool):
    cols = ["Modèle", "Stratégie", "Variante", "Score", "Score CE", "Fichier", "Chunk", "Texte"]
    if not query.strip():
        return pd.DataFrame(), ""
    if not embedder.models:
        return pd.DataFrame([{"Erreur": "Chargez d'abord les modèles."}]), ""

    hyde_text = ""
    if hyde:
        hyde_text = embedder.generate_hyde(query)

    results = _get_searcher().search_embeddings(
        query, int(top_k), rerank=bool(rerank), n_candidates=int(n_candidates),
        hyde=bool(hyde), hyde_text=hyde_text,
    )
    df = pd.DataFrame(results) if results else pd.DataFrame(columns=cols)
    return df, hyde_text


def search_fulltext(query: str, top_k: int):
    if not query.strip():
        return pd.DataFrame()
    if not _get_indexer().opensearch_available():
        return pd.DataFrame([{"Erreur": "OpenSearch non disponible. Lancez: docker compose up -d"}])
    results = _get_searcher().search_fulltext(query, int(top_k))
    return pd.DataFrame(results) if results else pd.DataFrame(columns=["Score", "Fichier", "Chunk", "Extraits"])


with gr.Blocks(
    title="Recherche Docs",
    # Hauteur fixe d'environ 20 lignes pour les tables de résultats.
    css=".tall-results .table-wrap { min-height: 700px; max-height: 700px; }",
) as demo:
    gr.Markdown("# Recherche dans des documents Markdown\nEmbeddings GPU + Full-text OpenSearch")

    with gr.Row():
        btn_load = gr.Button("Charger les modèles", variant="primary")
        btn_unload = gr.Button("Décharger les modèles", variant="secondary")
        btn_clear = gr.Button("Tout effacer", variant="stop")

    status_box = gr.Textbox(label="Statut", interactive=False, lines=2)

    with gr.Row():
        models_table = gr.Dataframe(
            label="Modèles d'embeddings",
            headers=["Modèle", "Dimensions", "Chargé", "Device"],
            interactive=False,
            value=pd.DataFrame(embedder.status()),
        )

    gr.Markdown("---")
    gr.Markdown("## Indexation")

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Document Markdown (.md)", file_types=[".md"])
        with gr.Column(scale=2):
            strategy_sel = gr.Dropdown(
                choices=list(CHUNKERS.keys()),
                value=list(CHUNKERS.keys())[0],
                label="Stratégie de découpage",
            )
        with gr.Column(scale=1):
            chunk_size_sl = gr.Slider(64, 1024, value=512, step=64, label="Taille chunk (mots)", visible=True)
            overlap_sl = gr.Slider(0, 256, value=64, step=16, label="Overlap (mots)", visible=True)

    btn_index = gr.Button("Indexer le document", variant="secondary")
    index_status = gr.Textbox(label="Résultat indexation", interactive=False, lines=4)
    indexed_table = gr.Dataframe(label="Collections Qdrant indexées", interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Recherche")

    with gr.Tabs():
        with gr.Tab("Embeddings"):
            gr.Markdown(
                "_La recherche interroge **toutes les collections** ; les résultats "
                "sont regroupés par collection (modèle × stratégie)._"
            )
            with gr.Row():
                emb_query = gr.Textbox(label="Requête", placeholder="Entrez votre question…", scale=4)
                emb_topk = gr.Slider(1, 20, value=5, step=1, label="Top K (par collection)", scale=1)
            with gr.Row():
                emb_rerank = gr.Checkbox(
                    value=False,
                    label="Activer le reranking (BAAI/bge-reranker-v2-m3)",
                    scale=2,
                )
                emb_candidates = gr.Slider(
                    5, 100, value=20, step=5,
                    label="Candidats avant rerank (par collection)",
                    scale=2,
                )
            with gr.Row():
                emb_hyde = gr.Checkbox(
                    value=False,
                    label="Activer HyDE (compare « requête seule » vs « HyDE », LLM Qwen2.5-7B)",
                    scale=4,
                )
            gr.Markdown(
                "_HyDE n'apporte en principe que pour le modèle **symétrique** "
                "`paraphrase-multilingual-mpnet-base-v2` ; les modèles E5 et bge-m3, "
                "orientés question→passage, n'en ont pas besoin._"
            )
            emb_hyde_text = gr.Textbox(
                label="Pseudo-document HyDE généré (encodé à la place de la question)",
                interactive=False, lines=3, visible=True,
            )
            btn_emb_search = gr.Button("Rechercher par embeddings")
            emb_results = gr.Dataframe(
                label="Résultats", interactive=False, elem_classes=["tall-results"]
            )

        with gr.Tab("Full-text (OpenSearch)"):
            gr.Markdown(
                'Syntaxe : `mot`, `mot*`, `"mot exact"`, `A AND B`, `A OR B`, `(A OR B) AND C`'
            )
            with gr.Row():
                ft_query = gr.Textbox(
                    label="Requête OpenSearch",
                    placeholder='exemple : "intelligence" AND (machine* OR apprenti*)',
                    scale=4,
                )
                ft_topk = gr.Slider(1, 20, value=5, step=1, label="Top K", scale=1)
            btn_ft_search = gr.Button("Rechercher full-text")
            ft_results = gr.Dataframe(
                label="Résultats", interactive=False, elem_classes=["tall-results"]
            )

    # Afficher/masquer les sliders selon la stratégie choisie (FIXED_SIZE_LABEL
    # est défini au niveau module et partagé avec la logique de nommage).
    def toggle_fixed_params(strategy):
        visible = strategy == FIXED_SIZE_LABEL
        return gr.update(visible=visible), gr.update(visible=visible)

    strategy_sel.change(toggle_fixed_params, inputs=[strategy_sel], outputs=[chunk_size_sl, overlap_sl])

    # Events
    # L'avancement n'apparaît que sur la sortie directe (Statut / Résultat) ;
    # les tables sont rafraîchies ensuite via .then().
    btn_load.click(load_models, outputs=[status_box]).then(
        _models_status_df, outputs=[models_table]
    )
    btn_unload.click(unload_models, outputs=[models_table, status_box])
    btn_clear.click(clear_all, outputs=[status_box, indexed_table])
    btn_index.click(
        index_document,
        inputs=[file_input, strategy_sel, chunk_size_sl, overlap_sl],
        outputs=[index_status],
    ).then(_collections_df, outputs=[indexed_table])
    btn_emb_search.click(
        search_embeddings,
        inputs=[emb_query, emb_topk, emb_rerank, emb_candidates, emb_hyde],
        outputs=[emb_results, emb_hyde_text],
    )
    btn_ft_search.click(
        search_fulltext,
        inputs=[ft_query, ft_topk],
        outputs=[ft_results],
    )

    # À chaque (re)chargement de la page : refléter l'état serveur courant
    # (modèles réellement chargés en mémoire + collections Qdrant existantes).
    demo.load(_refresh_on_load, outputs=[models_table, indexed_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
