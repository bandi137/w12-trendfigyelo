"""W12 – Kutatási trendfigyelő RAG-gal. A notebook ezeket a modulokat használja."""
from .data import (day_ranges, download_dates, download_month, load_or_download, month_ranges, per_day_quota,
                   per_month_limit, sampling_of)
from .embed import Embedder, load_or_embed
from .llm import generate
from .rag import RAG, retrieve, build_prompt, check_citations, detect_language, resolve_language
from .trends import (find_topics, topic_trends, name_topics, build_naming_prompt, parse_names,
                     trend_table, style_table, plot_trends, summarize_trends)
from .evaluate import hit_at_k, evaluate, first_rank, load_questions
from .evalviz import eval_figure, eval_overview, eval_view, load_scores, overview_text, rank_bin, scores_figure
from .report import build_report, trend_figure, md_lite
from .tmap import map_coords, map_question, question_position, topic_map
from .publish import publish_files, list_archive, pages_url, PublishError
from .app import TrendApp
from . import prompts
