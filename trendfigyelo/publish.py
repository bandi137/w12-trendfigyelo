"""HTML-jelentés közzététele a GitHub Pages-re: a fájlok egyetlen commitban kerülnek a repó `docs/` mappájába.

A GitHub REST API-t használja (git nélkül), egy személyes hozzáférési tokennel (Colab Secrets: GITHUB_TOKEN).
- Repótulajdonos: finomhangolt (fine-grained) token, csak erre a repóra, Contents: Read and write.
- Közreműködő: klasszikus token `public_repo` jogosultsággal (a finomhangolt token közreműködőként nem működik).
"""
import re

import requests

API = "https://api.github.com"


class PublishError(RuntimeError):
    pass


def pages_url(repo):
    owner, name = repo.split("/")
    return f"https://{owner.lower()}.github.io/{name}/"


def _session(token, session=None):
    s = session or requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                      "X-GitHub-Api-Version": "2022-11-28"})
    return s


def _check(resp, what):
    if resp.status_code >= 400:
        hint = {401: "érvénytelen vagy lejárt token", 403: "a tokennek nincs írási joga ehhez a repóhoz",
                404: "a repó vagy az ág nem található (vagy a token nem látja)"}.get(resp.status_code, "")
        raise PublishError(f"{what} sikertelen ({resp.status_code}{', ' + hint if hint else ''}): {resp.text[:200]}")
    return resp.json() if resp.content else {}


def list_archive(token, repo, branch="main", folder="docs", session=None):
    """A már közzétett archív jelentések dátumcímkéi (docs/archiv/*.html)."""
    s = _session(token, session)
    r = s.get(f"{API}/repos/{repo}/contents/{folder}/archiv", params={"ref": branch}, timeout=30)
    if r.status_code == 404:
        return []
    items = _check(r, "Az archívum listázása")
    return sorted(i["name"][:-5] for i in items if i.get("name", "").endswith(".html"))


def publish_files(files, token, repo, branch="main", message="Jelentés frissítése", session=None, retries=1,
                  delete=()):
    """files: {"docs/index.html": "<html>…", …}; delete: törlendő fájlok útvonalai. Egyetlen commitot hoz létre;
    ha közben valaki más is feltöltött, újrakezdi a legfrissebb állapotról. Visszaadja a commit adatait."""
    s = _session(token, session)
    for attempt in range(retries + 1):
        ref = _check(s.get(f"{API}/repos/{repo}/git/ref/heads/{branch}", timeout=30), "Az ág lekérése")
        head = ref["object"]["sha"]
        base_tree = _check(s.get(f"{API}/repos/{repo}/git/commits/{head}", timeout=30), "A commit lekérése")["tree"]["sha"]
        tree = _check(s.post(f"{API}/repos/{repo}/git/trees", timeout=60, json={
            "base_tree": base_tree,
            "tree": [{"path": p, "mode": "100644", "type": "blob", "content": c} for p, c in files.items()]
            + [{"path": p, "mode": "100644", "type": "blob", "sha": None} for p in delete]}),
            "A fájlok feltöltése")
        commit = _check(s.post(f"{API}/repos/{repo}/git/commits", timeout=30, json={
            "message": message, "tree": tree["sha"], "parents": [head]}), "A commit létrehozása")
        r = s.patch(f"{API}/repos/{repo}/git/refs/heads/{branch}", timeout=30, json={"sha": commit["sha"], "force": False})
        if r.status_code == 422 and attempt < retries:   # közben valaki más is feltöltött -> újra a friss állapotról
            continue
        _check(r, "Az ág frissítése")
        return {"sha": commit["sha"], "commit_url": f"https://github.com/{repo}/commit/{commit['sha']}",
                "pages_url": pages_url(repo)}
    raise PublishError("Az ág frissítése többszöri próbálkozásra sem sikerült.")


def fetch_file(token, repo, path, branch="main", session=None):
    """Egy fájl szövege a repóból (nyers tartalom, 1 MB fölött is), vagy None, ha nincs ilyen."""
    s = _session(token, session)
    r = s.get(f"{API}/repos/{repo}/contents/{path}", params={"ref": branch}, timeout=60,
              headers={"Accept": "application/vnd.github.raw+json"})
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        _check(r, f"A(z) {path} letöltése")
    return r.content.decode("utf-8")


def remove_archive_links(page, labels):
    """A jelentés „Korábbi jelentések” listájából kiveszi a megadott archív oldalakra mutató linkeket
    (a kezdőoldalon "archiv/…", az archív oldalakon "…" alakúak); ha a lista kiürül, a szakasz és a menüpont is
    kikerül."""
    for label in labels:
        page = re.sub(r'<li><a href="(?:archiv/)?' + re.escape(label) + r'\.html">[^<]*</a></li>', "", page)
    if re.search(r'<ul class="archive">\s*</ul>', page):
        page = re.sub(r'<section id="archiv">.*?</section>', "", page, flags=re.S)
        page = page.replace('<a href="#archiv">Archívum</a>', "")
    return page


def delete_archives(labels, token, repo, branch="main", folder="docs", session=None):
    """Archív jelentések törlése egyetlen commitban: a docs/archiv/<címke>.html fájlok törlése, és a kezdőoldal,
    valamint a megmaradó archív oldalak archívumlistájából a rájuk mutató linkek eltávolítása (halott link ne
    maradjon). A jelentések tartalmát nem számolja újra. Visszaadja a commit adatait és a törölt címkéket."""
    existing = set(list_archive(token, repo, branch, folder, session))
    labels = sorted(set(labels) & existing)
    if not labels:
        raise PublishError("Nincs törölhető archív jelentés a kijelöltek között (lehet, hogy közben már törölték).")
    files = {}
    pages = [f"{folder}/index.html"] + [f"{folder}/archiv/{a}.html" for a in sorted(existing - set(labels))]
    for path in pages:
        text = fetch_file(token, repo, path, branch, session)
        if text is None:
            continue
        new = remove_archive_links(text, labels)
        if new != text:
            files[path] = new
    res = publish_files(files, token, repo, branch, session=session,
                        message=f"Archív jelentések törlése: {', '.join(labels)}",
                        delete=[f"{folder}/archiv/{a}.html" for a in labels])
    return {**res, "deleted": labels, "updated": sorted(files)}
