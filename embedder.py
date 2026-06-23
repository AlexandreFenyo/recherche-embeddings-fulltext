import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
HYDE_LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"

MODELS_CONFIG = {
    "multilingual-e5-large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "dims": 1024,
        "needs_prefix": True,
    },
    "bge-m3": {
        "hf_name": "BAAI/bge-m3",
        "dims": 1024,
        "needs_prefix": False,
    },
    "multilingual-e5-base": {
        "hf_name": "intfloat/multilingual-e5-base",
        "dims": 768,
        "needs_prefix": True,
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "hf_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dims": 768,
        "needs_prefix": False,
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    def __init__(self):
        self.models: dict[str, SentenceTransformer] = {}
        self.reranker: CrossEncoder | None = None
        self._hyde_tok = None
        self._hyde_model = None

    def get_reranker(self) -> CrossEncoder:
        """Charge le cross-encoder de reranking à la demande (paresseux)."""
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANKER_NAME, device=DEVICE)
        return self.reranker

    def _get_hyde(self):
        """Charge le LLM génératif HyDE à la demande (paresseux)."""
        if self._hyde_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._hyde_tok = AutoTokenizer.from_pretrained(HYDE_LLM_NAME)
            self._hyde_model = AutoModelForCausalLM.from_pretrained(
                HYDE_LLM_NAME,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(DEVICE)
            self._hyde_model.eval()
        return self._hyde_tok, self._hyde_model

    def generate_hyde(self, query: str, max_new_tokens: int = 220) -> str:
        """HyDE : génère un pseudo-document (passage hypothétique) répondant à la
        requête, qu'on encodera à la place (ou en plus) de la question brute."""
        tok, model = self._get_hyde()
        messages = [
            {
                "role": "system",
                "content": "Tu es un expert qui rédige des extraits de documents techniques en français.",
            },
            {
                "role": "user",
                "content": (
                    "Rédige un court passage (3 à 4 phrases, en français) qui pourrait "
                    "être extrait d'un document et qui répondrait à la question suivante. "
                    "Écris uniquement le passage, sans préambule ni guillemets.\n\n"
                    f"Question : {query}"
                ),
            },
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    def load_all(self, progress_cb=None):
        for name, cfg in MODELS_CONFIG.items():
            if name not in self.models:
                if progress_cb:
                    progress_cb(f"Chargement {name}…")
                self.models[name] = SentenceTransformer(cfg["hf_name"], device=DEVICE)
        if progress_cb:
            progress_cb("Tous les modèles chargés.")

    def unload_all(self):
        """Décharge tous les modèles (embeddings + reranker + LLM HyDE) et rend la VRAM."""
        import gc

        self.models.clear()
        self.reranker = None
        self._hyde_tok = None
        self._hyde_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed_texts(self, model_name: str, texts: list[str], mode: str = "passage") -> np.ndarray:
        """mode: 'passage' for indexing, 'query' for search queries."""
        cfg = MODELS_CONFIG[model_name]
        model = self.models[model_name]

        if cfg["needs_prefix"]:
            prefix = "query: " if mode == "query" else "passage: "
            texts = [prefix + t for t in texts]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def status(self) -> list[dict]:
        rows = []
        for name, cfg in MODELS_CONFIG.items():
            rows.append({
                "Modèle": name,
                "Dimensions": cfg["dims"],
                "Chargé": "✓" if name in self.models else "—",
                "Device": DEVICE,
            })
        return rows
