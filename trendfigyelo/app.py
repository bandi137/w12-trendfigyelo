"""Gradio-felület a trendfigyelőhöz: kérdezés, trendek, kiértékelés és közzététel kód nélkül.

A notebook 7. szakasza indítja:  app = tf.TrendApp(...); app.launch()
"""
import os
import re
import subprocess
from datetime import datetime

from . import report
from .evaluate import evaluate, load_questions
from .data import download_dates
from .publish import PublishError, list_archive, pages_url, publish_files
from .tmap import PULL, map_coords, topic_map
from .trends import find_topics, name_topics, summarize_trends, topic_trends, trend_table

AUTHORS = "Sümegi András, Süle Szilvia"
REPO = "bandi137/w12-trendfigyelo"


def colab_secret(name):
    """A Colab Secrets fülről olvas; ha nincs ilyen titok vagy nincs hozzáférés, None."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def git_version(repo_dir):
    try:
        return subprocess.run(["git", "-C", repo_dir, "log", "-1", "--format=%h"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


class TrendApp:
    def __init__(self, df, emb, embedder, llm, rag, data_dir, settings, repo_dir="/content/repo", repo=REPO,
                 token_fn=lambda: colab_secret("GITHUB_TOKEN")):
        self.df, self.emb, self.embedder, self.llm, self.rag = df, emb, embedder, llm, rag
        self.data_dir, self.repo_dir, self.repo, self.token_fn = data_dir, repo_dir, repo, token_fn
        self.s = dict(settings)          # category, months, emb_model, llm_model, n_topics, n_show, top_k
        self.trends, self.summary, self.eval_df, self.eval_summary = None, "", None, None
        self.coords = None               # a tématérkép koordinátái (a Drive-on is tárolva)

    # --- műveletek (a felülettől függetlenül is hívhatók és tesztelhetők) ---------------------------

    def ask(self, question, k=None, month_from=None):
        question = (question or "").strip()
        if not question:
            return "Írj be egy kérdést."
        answer = self.rag.ask(question, k=k or self.s["top_k"], month_from=month_from or None, show=False)
        entry = self.rag.log[-1]
        hits = self.df.set_index("id").loc[[i for i in entry["sources"].split(", ") if i]]
        linked = re.sub(r"\[(\d{4}\.\d{4,5})\]", lambda m: f"[[{m.group(1)}]]({report.arxiv_url(m.group(1))})", answer)
        sources = "\n".join(f"{n}. [{i}]({report.arxiv_url(i)}) – {r.title} ({r.month})"
                            for n, (i, r) in enumerate(hits.iterrows(), 1))
        warn = f"\n\n> ⚠ A modell nem kapott forrásra is hivatkozott: {', '.join(entry['invalid'])}" if entry["invalid"] else ""
        sources = sources or "_A megadott időszakban nincs cikk._"
        return f"{linked}{warn}\n\n**Források**\n\n{sources}"

    def compute_trends(self, n_topics=None):
        n_topics = int(n_topics or self.s["n_topics"])
        df = self.df
        df["topic"], keywords = find_topics(self.emb, (df.title + ". " + df.abstract).values, n_topics=n_topics)
        share, growth = topic_trends(df.month, df.topic)
        ids = [int(t) for t in growth.index]
        names = name_topics(self.llm, self.s["llm_model"], df, keywords, ids, self.s["category"], self.data_dir)
        table = trend_table(df, ids, names, keywords, growth)
        table.to_csv(os.path.join(self.data_dir, "trendek.csv"), index=False)
        self.trends = dict(ids=ids, names=names, keywords=keywords, share=share, growth=growth, table=table)
        self.summary = ""                # a régi összefoglaló már nem a mostani témákról szól
        return table

    def figure(self, n=None):
        """A grafikon alapból annyi témát mutat, amennyi a táblázatban is van (n_show)."""
        t = self.trends
        return report.trend_figure(t["share"], t["growth"], t["names"], t["ids"], self.s["category"],
                                   n=int(n or self.s["n_show"]))

    def map_figure(self, play_button=False, pull=None):
        """Tématérkép a mostani témákkal. A koordináták (felügyelet nélküli UMAP) csak a vektoroktól függenek: elsőre
        1–2 perc, utána a Drive-ról töltődnek. pull: a témák összehúzásának erőssége (0–0,8; lásd tmap.PULL), a
        jelentés is ezt használja. play_button: csak a HTML-jelentésben működik."""
        if self.trends is None:
            self.compute_trends()
        if self.coords is None:
            self.coords = map_coords(self.emb, self.df.id, self.data_dir)
        if pull is not None:
            self.s["map_pull"] = float(pull)
        t = self.trends
        return topic_map(self.coords, self.df, t["names"], t["growth"], self.s["category"], play_button=play_button,
                         pull=self.s.get("map_pull", PULL))

    def summarize(self):
        if self.trends is None:
            self.compute_trends()
        t = self.trends
        self.summary = summarize_trends(self.llm, self.s["llm_model"], t["names"], t["growth"], t["ids"],
                                        self.s["category"], n=self.s["n_show"])
        return self.summary

    def run_eval(self, path=None):
        path = path or os.path.join(self.repo_dir, "eval", "kerdesek.csv")
        questions = load_questions(path, known_ids=set(self.df.id))
        if not questions:
            self.eval_df, self.eval_summary = None, None
            return None, {}
        self.eval_df, summary = evaluate(questions, lambda q, k: self.rag.retrieve(q, k=k))
        self.eval_summary = summary.to_dict()
        self.eval_df.to_csv(os.path.join(self.data_dir, "eval.csv"), index=False)
        return self.eval_df, self.eval_summary

    def meta(self):
        return dict(category=self.s["category"], n_papers=len(self.df),
                    months=f"{self.df.month.min()} – {self.df.month.max()}",
                    downloaded=download_dates(self.data_dir, sorted(self.df.month.unique())), emb_model=self.s["emb_model"],
                    llm_model=self.s["llm_model"], code_version=git_version(self.repo_dir),
                    generated=datetime.now().strftime("%Y-%m-%d %H:%M"), authors=AUTHORS,
                    repo_url=f"https://github.com/{self.repo}")

    def build_html(self, archive=None, archive_prefix="archiv/"):
        if self.trends is None:
            self.compute_trends()
        if not self.summary:
            self.summarize()
        return report.build_report(self.meta(), self.trends["table"], self.figure(), self.summary, self.rag.log,
                                   self.df, self.eval_df, self.eval_summary, archive, self.s["n_show"], archive_prefix,
                                   map_fig=self.map_figure(play_button=True))

    def preview(self):
        """A jelentés mentése a Drive-ra (letölthető előnézet), közzététel nélkül."""
        path = os.path.join(self.data_dir, "jelentes_elonezet.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_html())
        return path

    def publish(self):
        token = self.token_fn()
        if not token:
            raise PublishError("Nincs GITHUB_TOKEN a Colab Secrets fülön (vagy nincs hozzá notebook-hozzáférés). "
                               "Útmutató: docs/KOZZETETEL.md a repóban.")
        label = datetime.now().strftime("%Y-%m-%d")
        archive = sorted(set(list_archive(token, self.repo)) | {label})
        files = {"docs/index.html": self.build_html(archive, "archiv/"),
                 f"docs/archiv/{label}.html": self.build_html(archive, ""),
                 "docs/.nojekyll": ""}
        res = publish_files(files, token, self.repo, message=f"Jelentés közzététele: {label} ({self.s['category']})")
        with open(os.path.join(self.data_dir, f"jelentes_{label}.html"), "w", encoding="utf-8") as f:
            f.write(files["docs/index.html"])
        return res

    # --- felület -----------------------------------------------------------------------------------

    def build_ui(self):
        import gradio as gr

        def safe(fn):
            def wrapped(*a):
                try:
                    return fn(*a)
                except PublishError as e:
                    raise gr.Error(str(e))
                except Exception as e:
                    raise gr.Error(f"{type(e).__name__}: {e}")
            return wrapped

        months = sorted(self.df.month.unique())
        view_cols = ["név", "leírás", "kulcsszavak", "cikkek", "növekedés"]

        with gr.Blocks(title="Kutatási trendfigyelő") as ui:
            gr.Markdown(f"# Kutatási trendfigyelő – arXiv {self.s['category']}\n"
                        f"{len(self.df)} cikk, {months[0]} – {months[-1]} · nyelvi modell: `{self.s['llm_model']}`")
            with gr.Tab("Kérdezz"):
                q = gr.Textbox(label="Kérdés (magyarul vagy angolul)", lines=2,
                               placeholder="Pl.: Milyen módszerekkel csökkentik mostanában a hallucinációt?")
                with gr.Row():
                    k = gr.Slider(1, max(10, int(self.s["top_k"])), value=int(self.s["top_k"]), step=1,
                                  label="Hány cikket kapjon a modell")
                    since = gr.Dropdown(["(mind)"] + months, value="(mind)", label="Csak ettől a hónaptól")
                ask_btn = gr.Button("Kérdezek", variant="primary")
                answer = gr.Markdown()
                ask_btn.click(safe(lambda qq, kk, mm: self.ask(qq, int(kk), None if mm == "(mind)" else mm)),
                              [q, k, since], answer)
                q.submit(safe(lambda qq, kk, mm: self.ask(qq, int(kk), None if mm == "(mind)" else mm)),
                         [q, k, since], answer)

            with gr.Tab("Tématérkép"):
                gr.Markdown("Minden pont egy cikk; a hasonló cikkek egymás mellé kerülnek, a témák egymásba folynak. "
                            "Hogy a témák kivehetők legyenek, a cikkek a témájuk közepe felé húzódnak: ennek erőssége "
                            "a csúszkával állítható (0 = a nyers felhő). A szín a cikk témájának növekedése (piros: nő, "
                            "kék: csökken). A térkép alatti csúszkával egy hónap cikkei emelhetők ki (a lejátszás a "
                            "közzétett jelentésben működik). Elsőre a térkép számítása 1–2 perc, utána a Drive-ról "
                            "töltődik. A **Témák számítása** gomb (Trendek fül) a térképet is frissíti.")
                with gr.Row():
                    map_pull = gr.Slider(0, 0.8, value=PULL, step=0.05, label="Témák összehúzása", scale=3)
                    map_btn = gr.Button("Térkép rajzolása", variant="primary", scale=1)
                map_plot = gr.Plot(show_label=False)
                map_btn.click(safe(lambda p: self.map_figure(pull=p)), map_pull, map_plot)
                map_pull.release(safe(lambda p: self.map_figure(pull=p)), map_pull, map_plot)

            with gr.Tab("Trendek"):
                with gr.Row():
                    nt0, ns0 = int(self.s["n_topics"]), int(self.s["n_show"])
                    n_topics = gr.Slider(min(5, nt0), max(40, nt0), value=nt0, step=1, label="Témák száma")
                    n_show = gr.Slider(min(3, ns0), max(15, ns0), value=ns0, step=1, label="Megjelenített témák")
                trend_btn = gr.Button("Témák számítása", variant="primary")
                plot = gr.Plot()
                table = gr.Dataframe(wrap=True, interactive=False, column_widths=["22%", "43%", "21%", "7%", "7%"])
                sum_btn = gr.Button("Összefoglaló írása")
                summary = gr.Markdown()

                def do_trends(nt, ns):
                    self.s["n_show"] = int(ns)
                    t = self.compute_trends(int(nt))
                    return self.figure(), t[view_cols].head(int(ns)), "", self.map_figure()

                def do_show(ns):
                    """A megjelenített témák számának változtatása nem számol újra: ugyanazokból a témákból mutat többet/kevesebbet."""
                    self.s["n_show"] = int(ns)
                    if self.trends is None:
                        return gr.update(), gr.update()
                    return self.figure(), self.trends["table"][view_cols].head(int(ns))
                trend_btn.click(safe(do_trends), [n_topics, n_show], [plot, table, summary, map_plot])
                n_show.release(safe(do_show), n_show, [plot, table])
                sum_btn.click(safe(self.summarize), None, summary)

            with gr.Tab("Kiértékelés"):
                gr.Markdown("A tesztkérdések a repó `eval/kerdesek.csv` fájljában vannak. Szerkesztés után futtasd "
                            "újra a notebook első celláját, hogy a friss fájl lejöjjön a GitHubról.")
                eval_btn = gr.Button("Kiértékelés futtatása", variant="primary")
                eval_sum = gr.Markdown()
                eval_tab = gr.Dataframe(wrap=True, interactive=False)

                def do_eval():
                    res, summ = self.run_eval()
                    if res is None:
                        return "Még nincs tesztkérdés az `eval/kerdesek.csv`-ben.", None
                    return " · ".join(f"**{k}:** {v:.2f}" for k, v in summ.items()), res
                eval_btn.click(safe(do_eval), None, [eval_sum, eval_tab])

            with gr.Tab("Közzététel"):
                gr.Markdown(f"A jelentés (trendek, összefoglaló, az itt feltett kérdések és válaszok, kiértékelés) "
                            f"statikus weboldalként kerül ide: **{pages_url(self.repo)}**. Ha még nincsenek "
                            f"kiszámolva a trendek vagy nincs összefoglaló, a gomb elkészíti őket. A GitHub Pages "
                            f"1–2 perc alatt frissül.")
                with gr.Row():
                    prev_btn = gr.Button("Előnézet (letöltés)")
                    pub_btn = gr.Button("Közzététel", variant="primary")
                prev_file = gr.File(label="Előnézet")
                status = gr.Markdown()
                prev_btn.click(safe(self.preview), None, prev_file)

                def do_publish():
                    r = self.publish()
                    return (f"✅ Közzétéve. Weboldal: [{r['pages_url']}]({r['pages_url']}) · "
                            f"commit: [{r['sha'][:7]}]({r['commit_url']})")
                pub_btn.click(safe(do_publish), None, status)
        return ui

    def launch(self, share=True, password=None, user="w12", **kwargs):
        """A felület indítása. A nyilvános gradio.live linket jelszó védi: a jelszó a Colab Secrets `UI_PASSWORD`
        titka (vagy a `password` paraméter). Nyilvános linkkel (share=True) jelszó nélkül nem indul el, mert a
        linket ismerő bárki közzétehetne a GITHUB_TOKEN-nel és használhatná a Gemini API-keretet.
        Helyi indításnál (share=False) jelszó nélkül is elindul."""
        password = password or colab_secret("UI_PASSWORD")
        if password:
            kwargs.setdefault("auth", (user, password))
            kwargs.setdefault("auth_message", "W12 trendfigyelő – add meg a felhasználónevet és a jelszót.")
            print(f"A felület jelszóval védett (felhasználónév: {user}).")
        elif share:
            raise RuntimeError("A nyilvános felület jelszó nélkül nem indul el. Adj meg egy UI_PASSWORD titkot a "
                               "Colab 🔑 Secrets fülön (és engedélyezd a notebooknak), majd futtasd újra ezt a cellát.")
        self.ui = self.build_ui()
        return self.ui.launch(share=share, **kwargs)
