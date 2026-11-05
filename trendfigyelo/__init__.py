"""W12 – Kutatási trendfigyelő RAG-gal. A notebook ezeket a modulokat használja."""
from .data import month_ranges, load_or_download
from .embed import Embedder, load_or_embed
from .llm import generate
from .rag import RAG, retrieve, build_prompt, check_citations
from .trends import (find_topics, topic_trends, name_topics, build_naming_prompt, parse_names,
                     trend_table, style_table, plot_trends, summarize_trends)
from .evaluate import hit_at_k, evaluate
from . import prompts
