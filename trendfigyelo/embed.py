"""Szövegek vektorizálása (E5 modellcsalád: 'passage:' / 'query:' előtagokkal)."""
import os

import numpy as np


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def passages(self, texts, batch_size=64):
        return self.model.encode(["passage: " + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True)

    def query(self, question):
        return self.model.encode(["query: " + question], normalize_embeddings=True)[0]


def load_or_embed(df, embedder, data_dir):
    """Ha már van embeddings.npy a Drive-on, azt tölti be; különben kiszámolja és elmenti."""
    path = os.path.join(data_dir, "embeddings.npy")
    if os.path.exists(path):
        emb = np.load(path)
    else:
        emb = embedder.passages((df.title + ". " + df.abstract).tolist())
        np.save(path, emb)
    assert len(emb) == len(df), "A cikkek és a vektorok száma eltér – töröld az embeddings.npy-t és futtasd újra."
    return emb
