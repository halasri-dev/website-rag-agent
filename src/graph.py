"""LangGraph RAG workflow: retrieve -> (generate | refuse)"""
import json
import os
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

from usage import log_usage

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_DIR = "data/chroma_db"
COLLECTION = "litestream_docs"
FETCH_K = 20         # candidate chunks to look at
TOP_K = 8            # chunks sent to the LLM
MAX_PER_PAGE = 3     # so one page can't crowd out the others
MAX_DISTANCE = 0.9   # farther than this = clearly off-topic, refuse without calling the LLM
REFUSAL = "I couldn't find enough information on the Litestream website to answer that."

MODEL_NAME = os.environ["GEMINI_MODEL"]
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=os.environ["GOOGLE_API_KEY"])
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
)
db = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=DB_DIR)

SYSTEM_PROMPT = """You answer questions about the Litestream documentation using ONLY the numbered context passages you are given.

Rules:
1. Use only the context passages. Never use outside knowledge, even if you know the answer.
2. Find "evidence": the exact sentence(s) copied from the passages that directly answer the question. If there is none, evidence must be an empty string and enough_info must be false. When the question has several parts or needs several passages, copy the key sentence from each and combine them.
3. A passage that merely mentions a term does NOT answer a question about it. Comparisons, examples and side remarks are not evidence. Example: if a passage says "X works unlike Oracle", the question "Does X support Oracle?" is NOT answered.
4. If the passages answer only PART of the question, set enough_info to true, answer the part that is supported, and end with one sentence saying which part the passages do not cover.
5. If the question contains a false claim or assumption, say so briefly using the passages, then answer the supported part (enough_info true if the passages directly correct it). If the passages give nothing relevant, set enough_info to false.
6. Never answer "No" or "Yes" based on absence or on a passing mention alone.
7. When enough_info is false, use "answer" for one short sentence saying what the passages do cover instead, or which part of the question they do not support.
8. Be concise and accurate. Keep commands, option names and values exactly as written in the passages.
9. Reply with ONLY a JSON object, no other text, with the keys in this order:
{"evidence": "copied sentence(s) or empty string", "enough_info": true or false, "answer": "your answer", "used_passages": [numbers of the passages you used]}"""


class State(TypedDict, total=False):
    question: str
    hits: list
    best_distance: float
    answer: str
    sources: list
    evidence: str
    enough_info: bool
    input_tokens: int
    output_tokens: int


def call_llm(messages, tries=4):
    """Call Gemini; wait and retry on busy (503) or per-minute rate-limit (429) errors."""
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:  # daily quota used up: waiting won't help
                raise RuntimeError(
                    f"Daily free-tier quota used up for {MODEL_NAME}. "
                    "Wait for the reset or switch GEMINI_MODEL."
                ) from e
            transient = any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == tries - 1:
                raise
            wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
            print(f"  (Gemini busy, retrying in {wait}s...)")
            time.sleep(wait)


def text_of(msg):
    """Gemini may return a string or a list of content blocks; get the text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(m.group(0)) if m else None


def sub_queries(question):
    """Split 'A and B' questions so each half gets its own search."""
    parts = [p.strip(" ?.,") for p in re.split(r"\band\b|;", question, flags=re.IGNORECASE)]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts[:3] if len(parts) > 1 else []


def passages_supporting(evidence, hits):
    """Passage numbers whose text actually contains the model's quoted evidence."""
    norm = lambda s: re.sub(r"\s+", " ", s).lower().strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", str(evidence)) if len(s.strip()) > 25]
    found = []
    for i, (doc, _) in enumerate(hits, 1):
        text = norm(doc.page_content)
        if any(norm(s) in text for s in sentences):
            found.append(i)
    return found


# ---- graph nodes -------------------------------------------------------
def retrieve(state):
    question = state["question"]
    main = db.similarity_search_with_score(question, k=FETCH_K)
    best = min((s for _, s in main), default=9.0)  # off-topic routing uses the full question only

    # priority order: top 2 chunks of each sub-question first, then the main ranking
    candidates = []
    for part in sub_queries(question):
        candidates += db.similarity_search_with_score(part, k=2)
    candidates += main

    seen, hits, per_page = set(), [], {}
    for doc, score in candidates:
        cid, url = doc.metadata["chunk_id"], doc.metadata["url"]
        if cid in seen or per_page.get(url, 0) >= MAX_PER_PAGE:
            continue
        seen.add(cid)
        per_page[url] = per_page.get(url, 0) + 1
        hits.append((doc, score))
        if len(hits) == TOP_K:
            break
    return {"hits": hits, "best_distance": best}


def route(state):
    return "generate" if state["best_distance"] <= MAX_DISTANCE else "refuse"


def generate(state):
    hits = state["hits"]
    context = "\n\n".join(f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(hits, 1))
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context passages:\n{context}\n\nQuestion: {state['question']}"),
    ]

    in_tok = out_tok = 0
    data = None
    for _ in range(2):  # one retry if the JSON comes back malformed
        resp = call_llm(messages)
        usage = resp.usage_metadata or {}
        in_tok += usage.get("input_tokens", 0)
        out_tok += usage.get("output_tokens", 0)
        try:
            data = parse_json(text_of(resp))
        except json.JSONDecodeError:
            data = None
        if data is not None:
            break

    log_usage("query", MODEL_NAME, in_tok, out_tok, note=state["question"][:80])

    evidence = str((data or {}).get("evidence", "")).strip()
    if not data or not data.get("enough_info") or not evidence:
        note = (data or {}).get("answer", "").strip()
        return {"answer": f"{REFUSAL} {note}".strip(), "sources": [], "evidence": evidence,
                "enough_info": False, "input_tokens": in_tok, "output_tokens": out_tok}

    # Sources: passages the model reported PLUS passages verified to contain its evidence text.
    # URLs always come from chunk metadata, never typed by the LLM.
    verified = passages_supporting(evidence, hits)
    reported = [i for i in data.get("used_passages", []) if isinstance(i, int) and 1 <= i <= len(hits)]
    used = list(dict.fromkeys(verified + reported))
    urls = []
    for i in used or [1]:
        url = hits[i - 1][0].metadata["url"]
        if url not in urls:
            urls.append(url)
    return {"answer": data.get("answer", "").strip(), "sources": urls, "evidence": evidence,
            "enough_info": True, "input_tokens": in_tok, "output_tokens": out_tok}


def refuse(state):
    return {"answer": REFUSAL, "sources": [], "evidence": "", "enough_info": False,
            "input_tokens": 0, "output_tokens": 0}


# ---- build the graph ---------------------------------------------------
_g = StateGraph(State)
_g.add_node("retrieve", retrieve)
_g.add_node("generate", generate)
_g.add_node("refuse", refuse)
_g.add_edge(START, "retrieve")
_g.add_conditional_edges("retrieve", route, {"generate": "generate", "refuse": "refuse"})
_g.add_edge("generate", END)
_g.add_edge("refuse", END)
app = _g.compile()


def ask(question):
    return app.invoke({"question": question})