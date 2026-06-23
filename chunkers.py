import re
import numpy as np


class FixedSizeChunker:
    """Split text into overlapping word-based windows."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = max(1, int(chunk_size))
        # L'overlap doit rester strictement inférieur à la taille du chunk,
        # sinon la fenêtre n'avance jamais (boucle infinie).
        self.overlap = max(0, min(int(overlap), self.chunk_size - 1))

    def chunk(self, text: str) -> list[dict]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap  # garanti >= 1 par __init__
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"text": chunk_text, "metadata": {"start_word": start, "end_word": end}})
            if end == len(words):
                break
            start += step
        return chunks


class MarkdownParagraphChunker:
    """Split on double newlines and Markdown headers, merging small fragments."""

    def __init__(self, min_chars: int = 100, max_chars: int = 2000):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[dict]:
        parts = re.split(r'\n(?=#{1,6}\s)|\n\n+', text)
        parts = [p.strip() for p in parts if p.strip()]

        merged = []
        buffer = ""
        for part in parts:
            if not buffer:
                buffer = part
            elif len(buffer) + len(part) + 2 <= self.max_chars and len(buffer) < self.min_chars:
                buffer += "\n\n" + part
            else:
                merged.append(buffer)
                buffer = part
        if buffer:
            merged.append(buffer)

        # Redécoupe les blocs qui dépassent max_chars (un paragraphe unique trop
        # long n'était jamais scindé, produisant des chunks trop gros).
        out = []
        for block in merged:
            for piece in self._split_oversized(block):
                out.append(piece)

        return [{"text": t, "metadata": {"paragraph_index": i}} for i, t in enumerate(out)]

    def _split_oversized(self, block: str) -> list[str]:
        if len(block) <= self.max_chars:
            return [block]
        words = block.split()
        pieces, buf = [], ""
        for w in words:
            if buf and len(buf) + 1 + len(w) > self.max_chars:
                pieces.append(buf)
                buf = w
            else:
                buf = f"{buf} {w}" if buf else w
        if buf:
            pieces.append(buf)
        return pieces


class SpacySentenceChunker:
    """Sentence segmentation using spaCy French model."""

    def __init__(self, model: str = "fr_core_news_md", sentences_per_chunk: int = 3):
        self.model_name = model
        self.sentences_per_chunk = sentences_per_chunk
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                # Modèle français absent : repli sur un segmenteur de phrases minimal.
                self._nlp = spacy.blank("fr")
                self._nlp.add_pipe("sentencizer")

    def chunk(self, text: str) -> list[dict]:
        self._load()
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        chunks = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[i:i + self.sentences_per_chunk]
            chunks.append({
                "text": " ".join(group),
                "metadata": {"sentence_start": i, "sentence_end": i + len(group)},
            })
        return chunks


class SemanticChunker:
    """Group sentences by cosine similarity until a drop threshold is hit."""

    def __init__(self, embedder, threshold: float = 0.7, max_chunk_sentences: int = 10):
        self._embedder = embedder
        self.threshold = threshold
        self.max_chunk_sentences = max_chunk_sentences

    def chunk(self, text: str) -> list[dict]:
        import spacy
        try:
            nlp = spacy.load("fr_core_news_md")
        except OSError:
            nlp = spacy.blank("fr")
            nlp.add_pipe("sentencizer")

        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return []

        # Use the first available model for semantic splitting
        model_name = list(self._embedder.models.keys())[0]
        embeddings = self._embedder.embed_texts(model_name, sentences, mode="passage")

        groups = []
        current = [0]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if sim < self.threshold or len(current) >= self.max_chunk_sentences:
                groups.append(current)
                current = [i]
            else:
                current.append(i)
        groups.append(current)

        return [
            {
                "text": " ".join(sentences[j] for j in grp),
                "metadata": {"sentence_indices": grp},
            }
            for grp in groups
        ]


CHUNKERS = {
    "Taille fixe (512 mots, overlap 64)": lambda embedder: FixedSizeChunker(512, 64),
    "Paragraphes Markdown": lambda embedder: MarkdownParagraphChunker(),
    "Phrases spaCy (3 phrases/chunk)": lambda embedder: SpacySentenceChunker(),
    "Chunks sémantiques": lambda embedder: SemanticChunker(embedder),
}
