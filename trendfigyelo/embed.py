"""Szövegek vektorizálása (E5 modellcsalád: 'passage:' / 'query:' előtagokkal)."""
import hashlib
import json
import os

import numpy as np


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def passages(self, texts, batch_size=64):
        return self.model.encode(["passage: " + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True)

    def query(self, question):
        return self.model.encode(["query: " + question], normalize_embeddings=True)[0]


def _fingerprint(df, model_name):
    ids = "\n".join(df.id.astype(str))
    return dict(model=model_name, n=len(df), ids_sha1=hashlib.sha1(ids.encode()).hexdigest())


def load_or_embed(df, embedder, data_dir):
    """Ha az embeddings.npy pontosan ezekhez a cikkekhez (ebben a sorrendben) és ezzel a modellel készült,
    betölti; ha bármi változott (új hónap, más modell), az összes vektort újraszámolja és elmenti.
    Az ellenőrzés adatai az embeddings.json-ban vannak."""
    path = os.path.join(data_dir, "embeddings.npy")
    meta_path = os.path.join(data_dir, "embeddings.json")
    want = _fingerprint(df, getattr(embedder, "model_name", ""))
    have = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
    if have == want and os.path.exists(path):
        emb = np.load(path)
        print("Vektorok betöltve a Drive-ról:", emb.shape)
    else:
        reason = "nincs még vektor" if not os.path.exists(path) else \
            "más modell" if have and have.get("model") != want["model"] else "változott a cikklista"
        print(f"Vektorizálás ({reason}): {len(df)} cikk újraszámolása…")
        emb = embedder.passages((df.title + ". " + df.abstract).tolist())
        np.save(path, emb)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(want, f, indent=1)
    assert len(emb) == len(df), "A cikkek és a vektorok száma eltér – töröld az embeddings.npy-t és futtasd újra."
    return emb
