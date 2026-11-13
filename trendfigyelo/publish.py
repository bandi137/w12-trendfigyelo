"""HTML-jelentés közzététele a GitHub Pages-re: a fájlok egyetlen commitban kerülnek a repó `docs/` mappájába.

A GitHub REST API-t használja (git nélkül), egy személyes hozzáférési tokennel (Colab Secrets: GITHUB_TOKEN).
- Repótulajdonos: finomhangolt (fine-grained) token, csak erre a repóra, Contents: Read and write.
- Közreműködő: klasszikus token `public_repo` jogosultsággal (a finomhangolt token közreműködőként nem működik).
"""
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


def publish_files(files, token, repo, branch="main", message="Jelentés frissítése", session=None, retries=1):
    """files: {"docs/index.html": "<html>…", …}. Egyetlen commitot hoz létre; ha közben valaki más is
    feltöltött, újrakezdi a legfrissebb állapotról. Visszaadja a commit adatait."""
    s = _session(token, session)
    for attempt in range(retries + 1):
        ref = _check(s.get(f"{API}/repos/{repo}/git/ref/heads/{branch}", timeout=30), "Az ág lekérése")
        head = ref["object"]["sha"]
        base_tree = _check(s.get(f"{API}/repos/{repo}/git/commits/{head}", timeout=30), "A commit lekérése")["tree"]["sha"]
        tree = _check(s.post(f"{API}/repos/{repo}/git/trees", timeout=60, json={
            "base_tree": base_tree,
            "tree": [{"path": p, "mode": "100644", "type": "blob", "content": c} for p, c in files.items()]}),
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
