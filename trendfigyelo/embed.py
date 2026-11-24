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


def _content_keys(df):
    """Cikkenként egy ellenőrzőösszeg: azonosító + cím + absztrakt (ezekből készül a vektor). Ha bármelyik változik,
    a cikk vektora újraszámolódik."""
    return [hashlib.sha1(f"{i}\n{t}\n{a}".encode()).hexdigest()
            for i, t, a in zip(df.id.astype(str), df.title.astype(str), df.abstract.astype(str))]


def load_or_embed(df, embedder, data_dir):
    """Vektorok a cikkekhez, a Drive-on tárolva (embeddings.npy + embeddings.json).
    Az embeddings.json a modell nevét és cikkenként a tartalom ellenőrzőösszegét tárolja (azonosító + cím +
    absztrakt). Csak az új vagy tartalmában módosult cikkek vektorizálódnak újra, a többi a régi fájlból jön; más
    modellnél vagy régi formátumú (ellenőrzőösszeg nélküli) tárolásnál minden vektor újraszámolódik."""
    path = os.path.join(data_dir, "embeddings.npy")
    meta_path = os.path.join(data_dir, "embeddings.json")
    model = getattr(embedder, "model_name", "")
    keys = _content_keys(df)
    have = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
    old = {}
    if have and have.get("model") == model and "keys" in have and os.path.exists(path):
        old_emb = np.load(path)
        if len(old_emb) == len(have["keys"]):
            old = {k: old_emb[i] for i, k in enumerate(have["keys"])}
    missing = [i for i, k in enumerate(keys) if k not in old]
    if not missing and have and have.get("keys") == keys:
        emb = np.stack([old[k] for k in keys])
        print("Vektorok betöltve a Drive-ról:", emb.shape)
        return emb
    if missing:
        reason = ("nincs még vektor" if not os.path.exists(path) else
                  "más modell" if have and have.get("model") != model else
                  "régi formátumú tárolás (egyszeri)" if have and "keys" not in have else
                  "új vagy módosult cikkek")
        print(f"Vektorizálás ({reason}): {len(missing)} / {len(df)} cikk…")
        texts = (df.title + ". " + df.abstract).tolist()
        new = embedder.passages([texts[i] for i in missing])
        for i, v in zip(missing, new):
            old[keys[i]] = np.asarray(v)
    emb = np.stack([old[k] for k in keys]).astype(np.float32)
    np.save(path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dict(model=model, n=len(df), keys=keys), f)
    assert len(emb) == len(df), "A cikkek és a vektorok száma eltér – töröld az embeddings.npy-t és futtasd újra."
    return emb
