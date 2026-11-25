"""Gradio-felület a trendfigyelőhöz: kérdezés, trendek, kiértékelés és közzététel kód nélkül.

A notebook 7. szakasza indítja:  app = tf.TrendApp(...); app.launch()
"""
import os
import re
import subprocess
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

from . import report
from .evaluate import evaluate, load_questions
from .verify import as_list, claims_markdown, summary_counts
from .evalviz import eval_figure, eval_view, load_scores, overview_text, scores_figure
from .data import download_dates, per_month_limit, sampling_of
from .publish import PublishError, delete_archives, list_archive, pages_url, publish_files
from .tmap import PULL, map_coords, map_question, topic_map
from .trends import find_topics, growth_note, name_topics, summarize_trends, topic_trends, trend_table

LANG_CHOICES = {"automatikus": None, "magyar": "hu", "angol": "en"}   # a felület választója → rag.ask(lang=)

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


EVAL_HELP = """### Mit mér ez a fül?
A rendszer egy kérdésre két lépésben válaszol. Először **kikeresi** a cikkek közül a kérdéshez leginkább hasonlókat,
aztán ezek közül az első néhányat (alapból 5-öt) odaadja a nyelvi modellnek, amely ezekből írja meg a választ.
Ha a keresés nem hozza elő a jó cikket, a modell sem tud jó választ adni, ezért a keresést külön ellenőrizzük.

**Hogyan?** Tesztkérdéseket írtunk magyarul és angolul, és mindegyikhez kézzel kikerestük azt a cikket, amelyik
valóban válaszol rá. A gomb minden kérdést lefuttat, és megnézi, **hányadik helyen** hozza a keresés a jó cikket.

**Hogyan olvasd az ábrát?** Egy négyzet egy kérdés, a szám a jó cikk helye a találatok között.
Sötétkék: az első helyen van; kék: a 2–5. helyen (ezt még megkapja a modell); világoskék: a 6–10. helyen
(alapbeállításban már nem kapja meg); szürke: nincs az első 10 között. Minél több a sötét négyzet, annál jobb.
A szakirodalom ezt *hit@k*-nak hívja: a hit@5 például azt adja meg, a kérdések hányad részénél van a jó cikk
az első 5 találatban. Egy négyzetre mutatva látszik a kérdés."""

SCORES_HELP = """### A válaszok kézi pontozása
A keresés mellett a nyelvi modell válaszait is ellenőriztük: a pontozó a választ a modellnek átadott cikkek
absztraktjaival vetette össze, és 1–5 ponttal értékelte, mennyire pontos, teljes és a forrásokkal alátámasztott
(5: pontos és teljes, helyes hivatkozásokkal; alacsonyabb: hiányos válasz, hibás hivatkozás, vagy a kapott cikkek
nem elegendőek a válaszhoz). Ez az ábra a korábban pontozott válaszokat mutatja (`eval/qa_pontozas.csv`), a gomb
nem változtat rajta. Egy sávra mutatva látszik az indoklás."""


TIMEZONE = ZoneInfo("Europe/Budapest")   # a Colab órája UTC; a jelentés és az archívum időpontja helyi idő


def now_local():
    return datetime.now(TIMEZONE)


def archive_label(when, existing=()):
    """Az archív jelentés egyedi neve: ÉÉÉÉ-HH-NN_ÓÓPP (helyi idő), így az aznapi újabb közzététel nem írja felül
    a korábbit; ugyanabban a percben ismételt közzétételnél -2, -3… végződést kap."""
    label = when.strftime("%Y-%m-%d_%H%M")
    n, out = 2, label
    while out in existing:
        out, n = f"{label}-{n}", n + 1
    return out


class TrendApp:
    def __init__(self, df, emb, embedder, llm, rag, data_dir, settings, repo_dir="/content/repo", repo=REPO,
                 token_fn=lambda: colab_secret("GITHUB_TOKEN")):
        self.df, self.emb, self.embedder, self.llm, self.rag = df, emb, embedder, llm, rag
        self.data_dir, self.repo_dir, self.repo, self.token_fn = data_dir, repo_dir, repo, token_fn
        self.s = dict(settings)          # category, months, emb_model, llm_model, n_topics, n_show, top_k
        self.trends, self.summary, self.eval_df, self.eval_summary = None, "", None, None
        self.coords = None               # a tématérkép koordinátái (a Drive-on is tárolva)
        self.ui_log = []                 # a felületen (Kérdezz fül) feltett kérdések naplóbejegyzései
        self.qa_choice = {}              # naplóbeli sorszám -> közzétegyük-e (új kérdés: igen)

    # --- műveletek (a felülettől függetlenül is hívhatók és tesztelhetők) ---------------------------

    def ask(self, question, k=None, month_from=None, lang=None, verify=None):
        """lang: 'hu' / 'en' a válasz nyelve; None vagy 'auto': a kérdésből felismerve.
        verify: a hivatkozások tartalmi ellenőrzése (None: a RAG beállítása szerint)."""
        question = (question or "").strip()
        if not question:
            return "Írj be egy kérdést."
        answer = self.rag.ask(question, k=k or self.s["top_k"], month_from=month_from or None, show=False, lang=lang,
                              verify=verify)
        entry = self.rag.log[-1]
        self.ui_log.append(entry)
        hits = self.df.set_index("id").loc[[i for i in entry["sources"].split(", ") if i]]
        linked = re.sub(r"\[(\d{4}\.\d{4,5})\]", lambda m: f"[[{m.group(1)}]]({report.arxiv_url(m.group(1))})", answer)
        sources = "\n".join(f"{n}. [{i}]({report.arxiv_url(i)}) – {r.title} ({r.month})"
                            for n, (i, r) in enumerate(hits.iterrows(), 1))
        warn = f"\n\n> ⚠ A modell nem kapott forrásra is hivatkozott: {', '.join(entry['invalid'])}" if entry["invalid"] else ""
        if entry.get("uncited"):
            warn += ("\n\n> ⚠ A válasz egyetlen átadott forrásra sem hivatkozik, ezért az állításai a forrásokkal "
                     "nem ellenőrizhetők.")
        warn += claims_markdown(entry.get("claims"), entry.get("verify_error", ""))
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

    def map_figure(self, play_button=False, pull=None, questions=None, active_question=0):
        """Tématérkép a mostani témákkal. A koordináták (felügyelet nélküli UMAP) csak a vektoroktól függenek: elsőre
        1–2 perc, utána a Drive-ról töltődnek. pull: a témák összehúzásának erőssége (0–0,8; lásd tmap.PULL), a
        jelentés is ezt használja. play_button: csak a HTML-jelentésben működik. questions: kérdések a térképen
        (tmap.map_question elemei)."""
        if self.trends is None:
            self.compute_trends()
        if self.coords is None:
            self.coords = map_coords(self.emb, self.df.id, self.data_dir)
        if pull is not None:
            self.s["map_pull"] = float(pull)
        t = self.trends
        return topic_map(self.coords, self.df, t["names"], t["growth"], self.s["category"], play_button=play_button,
                         pull=self.s.get("map_pull", PULL), questions=questions, active_question=active_question)

    def has_questions(self):
        """Van-e a felületen feltett, forrással megválaszolt kérdés (a Tématérkép fül jelölőnégyzetéhez)."""
        return any(e.get("sources") for e in self.ui_log)

    def asked_questions(self, ui_only=False):
        """Forrással megválaszolt kérdések a térképhez (tmap.map_question). ui_only: csak a felületen feltettek
        (a Tématérkép fül); különben a közzétételre kiválasztottak (a jelentés, ahol a kérdés–válaszok is szerepelnek)."""
        return [map_question(self.rag, e) for e in (self.ui_log if ui_only else self.selected_log()) if e.get("sources")]

    # --- a közzéteendő kérdés–válaszok kiválasztása -------------------------------------------------

    def qa_choices(self):
        """A Közzététel fül listája: [(felirat, sorszám szövegként)], és a kijelöltek sorszámai. A felirat jelzi,
        honnan jött a kérdés (notebook / felület), és a hivatkozás-ellenőrzés eredményét, hogy a gyenge válasz
        kiszűrhető legyen. Az eddig nem látott kérdések alapból kijelöltek."""
        ui = {id(e) for e in self.ui_log}
        out = []
        for i, e in enumerate(self.rag.log):
            self.qa_choice.setdefault(i, True)
            cnt = summary_counts(as_list(e.get("claims")))
            flags = (["❗ nem kapott forrásra hivatkozik"] if e.get("invalid") else []) + \
                    (["❗ nincs hivatkozás"] if e.get("uncited") else []) + \
                    ([f"✅{cnt['igen']} ⚠️{cnt['részben']} ❌{cnt['nem']}"] if sum(cnt.values()) else [])
            origin = "felület" if id(e) in ui else "notebook"
            label = f"{i + 1}. {textwrap.shorten(e['question'], 90, placeholder='…')} ({origin}" + \
                (f"; {'; '.join(flags)})" if flags else ")")
            out.append((label, str(i)))
        return out, [str(i) for i in range(len(self.rag.log)) if self.qa_choice.get(i, True)]

    def select_qa(self, values):
        """A kijelölés rögzítése (a Közzététel fül jelölőnégyzetei): a sorszámok szövegként."""
        keep = {int(v) for v in (values or [])}
        for i in range(len(self.rag.log)):
            self.qa_choice[i] = i in keep

    def selected_log(self):
        """A közzétételre kijelölt kérdés–válaszok (alapból mind)."""
        return [e for i, e in enumerate(self.rag.log) if self.qa_choice.get(i, True)]

    def summarize(self):
        if self.trends is None:
            self.compute_trends()
        t = self.trends
        self.summary = summarize_trends(self.llm, self.s["llm_model"], t["names"], t["growth"], t["ids"],
                                        self.s["category"], n=self.s["n_show"])
        return self.summary

    def scores(self):
        """A kézzel pontozott válaszok a repóból (eval/qa_pontozas.csv), ha vannak."""
        return load_scores(os.path.join(self.repo_dir, "eval", "qa_pontozas.csv"))

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
                    per_month=per_month_limit(self.data_dir, sorted(self.df.month.unique()))
                    or int(self.df.month.value_counts().max()),
                    sampling=sampling_of(self.data_dir, sorted(self.df.month.unique())), emb_dim=int(self.emb.shape[1]),
                    months=f"{self.df.month.min()} – {self.df.month.max()}",
                    downloaded=download_dates(self.data_dir, sorted(self.df.month.unique())), emb_model=self.s["emb_model"],
                    llm_model=self.s["llm_model"], code_version=git_version(self.repo_dir),
                    generated=now_local().strftime("%Y-%m-%d %H:%M"), authors=AUTHORS,
                    repo_url=f"https://github.com/{self.repo}")

    def build_html(self, archive=None, archive_prefix="archiv/"):
        if self.trends is None:
            self.compute_trends()
        if not self.summary:
            self.summarize()
        return report.build_report(self.meta(), self.trends["table"], self.figure(), self.summary, self.selected_log(),
                                   self.df, self.eval_df, self.eval_summary, archive, self.s["n_show"], archive_prefix,
                                   map_fig=self.map_figure(play_button=True, questions=self.asked_questions() or None),
                                   scores=self.scores())

    def preview(self):
        """A jelentés mentése a Drive-ra (letölthető előnézet), közzététel nélkül."""
        path = os.path.join(self.data_dir, "jelentes_elonezet.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_html())
        return path

    def _token(self):
        token = self.token_fn()
        if not token:
            raise PublishError("Nincs GITHUB_TOKEN a Colab Secrets fülön (vagy nincs hozzá notebook-hozzáférés). "
                               "Útmutató: docs/KOZZETETEL.md a repóban.")
        return token

    def archive_list(self):
        """A közzétett archív jelentések címkéi (újabb elöl)."""
        return sorted(list_archive(self._token(), self.repo), reverse=True)

    def delete_archives(self, labels):
        """A kijelölt archív jelentések törlése a weboldalról (egy commit; a linkek is kikerülnek)."""
        return delete_archives(list(labels or []), self._token(), self.repo)

    def publish(self):
        token = self._token()
        existing = set(list_archive(token, self.repo))
        label = archive_label(now_local(), existing)
        archive = sorted(existing | {label})
        files = {"docs/index.html": self.build_html(archive, "archiv/"),
                 f"docs/archiv/{label}.html": self.build_html(archive, ""),
                 "docs/.nojekyll": ""}
        res = publish_files(files, token, self.repo, message=f"Jelentés közzététele: {label} ({self.s['category']})")
        with open(os.path.join(self.data_dir, f"jelentes_{label}.html"), "w", encoding="utf-8") as f:
            f.write(files["docs/index.html"])
        return {**res, "label": label}

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

        def busy(triggers, btn, fn, inputs, outputs, status, msg):
            """Lassú művelet visszajelzéssel: kattintáskor a gomb „⏳ …” feliratot kap és letiltódik, a `status`
            mezőben megjelenik, mi történik; a végén (hiba esetén is) a gomb visszaáll, és ha a status mezőben még a
            várakozó üzenet áll, eltűnik. triggers: események (pl. btn.click, q.submit), amelyek elindítják."""
            label = btn.value

            def start():
                return gr.update(value=f"⏳ {label}…", interactive=False), f"⏳ {msg}"

            def finish(cur):
                return gr.update(value=label, interactive=True), ("" if str(cur or "").startswith("⏳") else cur)
            for trig in triggers:
                trig(start, None, [btn, status], queue=False) \
                    .then(safe(fn), inputs, outputs) \
                    .then(finish, status, [btn, status], queue=False)

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
                    lang = gr.Radio(list(LANG_CHOICES), value="automatikus", label="Válasz nyelve")
                check = gr.Checkbox(value=bool(getattr(self.rag, "verify", True)),
                                    label="Hivatkozások tartalmi ellenőrzése",
                                    info="Egy második modellhívás megnézi, hogy a hivatkozott cikk absztraktja "
                                         "alátámasztja-e az állítást (néhány másodperccel lassabb).")
                ask_btn = gr.Button("Kérdezek", variant="primary")
                answer = gr.Markdown()
                gr.Markdown("_A feltett kérdések és a forrásaik a **Tématérkép** fülön is megnézhetők._")

            with gr.Tab("Tématérkép"):
                gr.Markdown("Minden pont egy cikk; a hasonló cikkek egymás mellé kerülnek, a témák egymásba folynak. "
                            "A tengelyeknek nincs jelentése és mértékegysége: a közelség hasonlóságot jelent, a "
                            "távoli foltok közti távolság nem mérőszám. "
                            "Hogy a témák kivehetők legyenek, a cikkek a témájuk közepe felé húzódnak: ennek erőssége "
                            "a csúszkával állítható (0 = a nyers felhő). A szín a cikk témájának növekedése (piros: nő, "
                            "kék: csökken). A térkép alatti csúszkával egy hónap cikkei emelhetők ki (a lejátszás a "
                            "közzétett jelentésben működik). Elsőre a térkép számítása 1–2 perc, utána a Drive-ról "
                            "töltődik. A **Témák számítása** gomb (Trendek fül) a térképet is frissíti. "
                            "A Kérdezz fülön feltett kérdések a jelölőnégyzettel tehetők a térképre: a ★ a kérdés, a "
                            "számozott karikák a modellnek átadott források; több kérdés közül a térkép bal felső "
                            "menüjében lehet választani.")
                with gr.Row():
                    map_pull = gr.Slider(0, 0.8, value=PULL, step=0.05, label="Témák összehúzása", scale=3)
                    with gr.Column(scale=1):
                        has_q = self.has_questions()
                        show_q = gr.Checkbox(value=has_q, interactive=has_q,
                                             label="A kérdés és a források a tématérképen",
                                             info="A Kérdezz fülön feltett kérdés után választható.")
                        map_btn = gr.Button("Térkép rajzolása", variant="primary")
                map_status = gr.Markdown()
                map_plot = gr.Plot(show_label=False)

                def draw_map(p, sq):
                    """A térkép a választott összehúzással; bejelölve a feltett kérdésekkel (a legutóbbi látszik, a
                    többi a térkép bal felső menüjéből választható)."""
                    qs = self.asked_questions(ui_only=True) if sq else []
                    return self.map_figure(pull=p, questions=qs or None, active_question="last")
                busy([map_btn.click, map_pull.release, show_q.input], map_btn, draw_map, [map_pull, show_q], map_plot,
                     map_status, "A térkép rajzolása folyamatban (elsőre a koordináták számítása 1–2 perc).")

                def do_ask(qq, kk, mm, ll, cc):
                    text = self.ask(qq, int(kk), None if mm == "(mind)" else mm, LANG_CHOICES.get(ll), verify=bool(cc))
                    ok = self.has_questions()
                    return text, gr.update(interactive=ok, value=ok)
                busy([ask_btn.click, q.submit], ask_btn, do_ask, [q, k, since, lang, check], [answer, show_q], answer,
                     "A források keresése és a válasz megírása folyamatban (a nyelvi modell válasza általában "
                     "5–30 másodperc, a hivatkozások ellenőrzésével valamivel több).")

            with gr.Tab("Trendek"):
                gr.Markdown("A témákat a cikkek vektorainak csoportosítása adja, a nevüket és a leírásukat nyelvi "
                            "modell írja. " + growth_note())
                with gr.Row():
                    nt0, ns0 = int(self.s["n_topics"]), int(self.s["n_show"])
                    n_topics = gr.Slider(min(5, nt0), max(40, nt0), value=nt0, step=1, label="Témák száma")
                    n_show = gr.Slider(min(3, ns0), max(15, ns0), value=ns0, step=1, label="Megjelenített témák")
                trend_btn = gr.Button("Témák számítása", variant="primary")
                trend_status = gr.Markdown()
                plot = gr.Plot()
                table = gr.Dataframe(wrap=True, interactive=False, column_widths=["22%", "43%", "21%", "7%", "7%"])
                sum_btn = gr.Button("Összefoglaló írása")
                summary = gr.Markdown()

                def do_trends(nt, ns, p, sq):
                    self.s["n_show"] = int(ns)
                    t = self.compute_trends(int(nt))
                    return self.figure(), t[view_cols].head(int(ns)), "", draw_map(p, sq)

                def do_show(ns):
                    """A megjelenített témák számának változtatása nem számol újra: ugyanazokból a témákból mutat többet/kevesebbet."""
                    self.s["n_show"] = int(ns)
                    if self.trends is None:
                        return gr.update(), gr.update()
                    return self.figure(), self.trends["table"][view_cols].head(int(ns))
                busy([trend_btn.click], trend_btn, do_trends, [n_topics, n_show, map_pull, show_q],
                     [plot, table, summary, map_plot],
                     trend_status, "A témák számítása folyamatban: csoportosítás, kulcsszavak, témanevek a nyelvi "
                     "modelltől és a tématérkép (néhány másodperctől 1–2 percig).")
                n_show.release(safe(do_show), n_show, [plot, table])
                busy([sum_btn.click], sum_btn, self.summarize, None, summary, summary,
                     "Az összefoglaló készül: a nyelvi modell a témák és a növekedési adatok alapján ír (általában "
                     "10–60 másodperc).")

            with gr.Tab("Kiértékelés"):
                gr.Markdown(EVAL_HELP)
                eval_btn = gr.Button("Kiértékelés futtatása", variant="primary")
                eval_sum = gr.Markdown()
                eval_plot = gr.Plot(show_label=False)
                eval_tab = gr.Dataframe(wrap=True, interactive=False, visible=False)

                def do_eval():
                    res, summ = self.run_eval()
                    if res is None:
                        return "Még nincs tesztkérdés az `eval/kerdesek.csv`-ben.", None, gr.update(visible=False)
                    return overview_text(res), eval_figure(res), gr.update(value=eval_view(res), visible=True)
                busy([eval_btn.click], eval_btn, do_eval, None, [eval_sum, eval_plot, eval_tab], eval_sum,
                     "A kiértékelés fut: minden tesztkérdésre keresés a cikkek között (néhány másodperc).")
                scores = self.scores()
                if scores is not None and len(scores):
                    gr.Markdown(SCORES_HELP)
                    gr.Plot(value=scores_figure(scores, title=False), show_label=False)

            with gr.Tab("Közzététel") as pub_tab:
                gr.Markdown(f"A jelentés (trendek, tématérkép, összefoglaló, a kiválasztott kérdések és válaszok, "
                            f"kiértékelés) statikus weboldalként kerül ide: **{pages_url(self.repo)}**. Ha még nincsenek "
                            f"kiszámolva a trendek vagy nincs összefoglaló, a gomb elkészíti őket. A GitHub Pages "
                            f"1–2 perc alatt frissül. Minden közzététel külön archív oldalt kap (dátum és időpont), "
                            f"így az aznapi újabb közzététel nem írja felül a korábbit.")
                choices, chosen = self.qa_choices()
                qa_pick = gr.CheckboxGroup(choices=choices, value=chosen, label="Közzéteendő kérdések és válaszok",
                                           info="A jelentésbe és a tématérképére csak a kijelöltek kerülnek. Zárójelben: "
                                                "honnan jött a kérdés, és a hivatkozás-ellenőrzés eredménye "
                                                "(✅ alátámasztott, ⚠️ részben, ❌ nem alátámasztott állítások száma).")
                with gr.Row():
                    all_btn = gr.Button("Mind", size="sm", scale=0, min_width=110)
                    none_btn = gr.Button("Egyik sem", size="sm", scale=0, min_width=110)
                with gr.Row():
                    prev_btn = gr.Button("Előnézet (letöltés)")
                    pub_btn = gr.Button("Közzététel", variant="primary")
                prev_file = gr.File(label="Előnézet")
                status = gr.Markdown()

                def refresh_qa():
                    c, v = self.qa_choices()
                    return gr.update(choices=c, value=v)
                pub_tab.select(refresh_qa, None, qa_pick)
                qa_pick.input(self.select_qa, qa_pick, None)
                all_btn.click(lambda: [v for _, v in self.qa_choices()[0]], None, qa_pick).then(self.select_qa, qa_pick, None)
                none_btn.click(lambda: [], None, qa_pick).then(self.select_qa, qa_pick, None)

                def do_preview(values):
                    self.select_qa(values)
                    return self.preview()
                busy([prev_btn.click], prev_btn, do_preview, qa_pick, prev_file, status,
                     "Az előnézet készül (ha kell, a témák és az összefoglaló is).")

                def do_publish(values):
                    self.select_qa(values)
                    r = self.publish()
                    return (f"✅ Közzétéve ({len(self.selected_log())} kérdés–válasszal). Weboldal: "
                            f"[{r['pages_url']}]({r['pages_url']}) · archív oldal: `{r['label']}` · "
                            f"commit: [{r['sha'][:7]}]({r['commit_url']})")
                busy([pub_btn.click], pub_btn, do_publish, qa_pick, status, status,
                     "Közzététel folyamatban: a jelentés elkészítése és feltöltése a GitHubra.")

                with gr.Accordion("Archív jelentések törlése", open=False):
                    gr.Markdown("A kijelölt archív jelentések lekerülnek a weboldalról, és a rájuk mutató linkek is "
                                "kikerülnek a kezdőoldal és a többi archív oldal „Korábbi jelentések” listájából (egy "
                                "commitban). A jelentések tartalma egyébként nem változik. A repó nyilvános, ezért a "
                                "törölt fájl a git-előzményekben továbbra is megtalálható.")
                    list_btn = gr.Button("Archívum listázása", size="sm", scale=0, min_width=180)
                    arch_pick = gr.CheckboxGroup(choices=[], label="Archív jelentések (újabb elöl)")
                    confirm = gr.Checkbox(value=False, label="Értem: a kijelölt jelentések lekerülnek a weboldalról.")
                    del_btn = gr.Button("Kijelöltek törlése", variant="stop")
                    del_status = gr.Markdown()

                    def do_list():
                        labels = self.archive_list()
                        return (gr.update(choices=[(report.archive_title(a), a) for a in labels], value=[]),
                                f"{len(labels)} archív jelentés." if labels else "Nincs archív jelentés.")
                    busy([list_btn.click], list_btn, do_list, None, [arch_pick, del_status], del_status,
                         "Az archívum lekérése a GitHubról.")

                    def do_delete(labels, ok):
                        if not labels:
                            return gr.update(), "Jelölj ki legalább egy archív jelentést.", gr.update()
                        if not ok:
                            return gr.update(), "A törléshez jelöld be: „Értem: a kijelölt jelentések lekerülnek…”.", \
                                gr.update()
                        r = self.delete_archives(labels)
                        rest = self.archive_list()
                        return (gr.update(choices=[(report.archive_title(a), a) for a in rest], value=[]),
                                f"🗑 Törölve: {', '.join(report.archive_title(a) for a in r['deleted'])}. "
                                f"Frissített oldalak: {len(r['updated'])} · commit: [{r['sha'][:7]}]({r['commit_url']}). "
                                "A weboldal 1–2 perc alatt frissül.", gr.update(value=False))
                    busy([del_btn.click], del_btn, do_delete, [arch_pick, confirm], [arch_pick, del_status, confirm],
                         del_status, "Törlés folyamatban: a fájlok törlése és a linkek frissítése a GitHubon.")
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
