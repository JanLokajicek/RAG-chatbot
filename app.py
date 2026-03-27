import csv
import hashlib
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)  # override=True přepíše hodnoty injektované Streamlitem ze secrets.toml

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from rag_app import (
    load_pdf,
    split_documents,
    build_vectorstore,
    load_vectorstore,
    build_retriever,
    PDF_PATH,
    CHROMA_DIR,
    RAG_PROMPT,
    TOP_K,
)

# .env má přednost (lokální vývoj); st.secrets jako fallback (Streamlit Cloud)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")

pdf_path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

CHROMA_BASE = Path(CHROMA_DIR)

EVAL_CSV = Path("evaluations.csv")
EVAL_COLUMNS = ["timestamp", "document", "question", "answer", "pages", "rating", "comment", "latency_s", "confidence"]


# ---------------------------------------------------------------------------
# Uvítací zpráva — automatická analýza dokumentů
# ---------------------------------------------------------------------------

def build_welcome_message(doc_info: list, chunks: list) -> str:
    """
    Pošle prvních 5000 znaků obsahu dokumentů do Claude,
    který vytvoří hierarchický přehled pojištění.
    """
    # Vezmi vzorky z celého dokumentu, ne jen začátek — záhlaví kapitol jsou na začátku,
    # konkrétní připojištění a podmínky jsou dál v textu.
    step = max(1, len(chunks) // 20)
    sampled = chunks[::step][:20]
    combined = "\n\n".join(c.page_content for c in sampled)[:8000]

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )
    response = llm.invoke([
        SystemMessage(content=(
            "Jsi extrakční nástroj. Odpovídáš VÝHRADNĚ odrážkovým seznamem.\n"
            "Tvá odpověď začíná znakem '-' a neobsahuje nic jiného — žádný nadpis, "
            "žádné intro, žádné závěrečné věty."
        )),
        HumanMessage(content=(
            "Z textu níže vytvoř hierarchický seznam pojištění.\n\n"
            "FORMÁT:\n"
            "- Povinné ručení\n"
            "  - limit plnění škody na zdraví\n"
            "  - limit plnění škody na majetku\n"
            "- Havarijní pojištění\n"
            "  - allrisk / živelní škody\n"
            "  - pojištění čelního skla\n\n"
            "Použij KONKRÉTNÍ názvy z textu. Maximálně 15 řádků.\n\n"
            f"Text:\n{combined}"
        )),
    ])

    # Pokud model nepoužil odrážky, přidáme je sami.
    # Řádky s odsazením → podkategorie (  -), ostatní → hlavní (-)
    lines = response.content.strip().splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            result.append(line)          # model odrážku použil — ponech
        elif line.startswith((" ", "\t")):
            result.append("  - " + stripped)   # odsazený řádek → podkategorie
        else:
            result.append("- " + stripped)     # normální řádek → hlavní typ
    overview = "\n".join(result)

    return (
        "Dobrý den! Mám k dispozici informace o těchto pojištěních:\n\n"
        + overview
        + "\n\nNa co se chcete zeptat?"
    )


# ---------------------------------------------------------------------------
# Volání LLM — retriever + synchronní invoke
# ---------------------------------------------------------------------------

def invoke_answer(retriever, question: str):
    """
    1. Retriever vyhledá relevantní chunky.
    2. Prompt se sestaví z kontextu a otázky.
    3. ChatAnthropic synchronně vrátí odpověď.
    Vrátí (answer: str, source_documents: list).
    """
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_text = RAG_PROMPT.format(context=context, question=question)

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )

    response = llm.invoke(prompt_text)
    return response.content, docs


# ---------------------------------------------------------------------------
# Helpers — chroma
# ---------------------------------------------------------------------------

def _paths_hash(pdf_paths: tuple) -> str:
    return hashlib.md5("|".join(sorted(pdf_paths)).encode()).hexdigest()


def _chroma_dir(paths_hash: str) -> str:
    return str(CHROMA_BASE / paths_hash)


def _cleanup_old_chroma(keep_hash: str):
    if not CHROMA_BASE.exists():
        return
    for d in CHROMA_BASE.iterdir():
        if d.is_dir() and d.name != keep_hash:
            try:
                shutil.rmtree(d)
            except PermissionError:
                pass


# ---------------------------------------------------------------------------
# Helpers — evaluace
# ---------------------------------------------------------------------------

def _clean_answer(text: str) -> str:
    import re
    text = re.sub(r'(\*\*|\*|`)', '', text)
    text = re.sub(r'\n+', ' | ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _format_pages(pages: list) -> str:
    return ", ".join(str(p) for p in pages) if pages else ""


def save_evaluation(question, answer, pages, rating, comment, document="", latency_s="", confidence=""):
    write_header = not EVAL_CSV.exists()
    with EVAL_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(EVAL_COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            document,
            question,
            _clean_answer(answer),
            _format_pages(pages),
            rating,
            comment,
            latency_s,
            confidence,
        ])


def load_eval_stats():
    if not EVAL_CSV.exists():
        return 0, 0.0, []
    with EVAL_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    total = len(rows)
    if total == 0:
        return 0, 0.0, []
    positive = sum(1 for r in rows if len(r) > 5 and r[5] == "👍")
    pct = round(positive / total * 100, 1)
    last20_ratings = [r[5] for r in rows[-20:] if len(r) > 5]
    return total, pct, last20_ratings


def clear_evaluations():
    with EVAL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(EVAL_COLUMNS)


# ---------------------------------------------------------------------------
# RAG chain — cache klíč = hash souborů, každý set má vlastní chroma složku
# ---------------------------------------------------------------------------

def compute_confidence(vectorstore, question: str, top_k: int = 3) -> float:
    """
    Vezme top_k L2 vzdálenosti z ChromaDB a převede na procenta (0–100).
    Normalizované vektory: L2 ∈ [0, 2] → 0 = identické, 2 = opačné.
    Převod: score = (1 - dist/2) * 100
    """
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    if not results:
        return 0.0
    scores = [max(0.0, (1 - dist / 2) * 100) for _, dist in results]
    return round(sum(scores) / len(scores), 1)


def show_confidence_badge(confidence: float):
    if confidence >= 90:
        color, label = "#1e7e34", f"✅ Vysoká jistota ({confidence} %)"
    elif confidence >= 70:
        color, label = "#856404", f"⚠️ Střední jistota ({confidence} %)"
    else:
        color, label = "#721c24", f"❌ Nízká jistota — ověřte zdroj ({confidence} %)"
    st.markdown(
        f'<span style="color:{color};font-size:0.85em;font-weight:600">{label}</span>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Načítám PDF a stavím RAG chain…")
def get_retriever(paths_hash: str, pdf_paths: tuple):
    all_chunks, doc_info, total_pages = [], [], 0
    for p in pdf_paths:
        pages = load_pdf(p)
        chunks = split_documents(pages)
        all_chunks.extend(chunks)
        doc_info.append((Path(p).name, len(pages)))
        total_pages += len(pages)

    persist_dir = _chroma_dir(paths_hash)
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        vectorstore = load_vectorstore(persist_dir)
    else:
        vectorstore = build_vectorstore(all_chunks, persist_dir)

    retriever = build_retriever(vectorstore, all_chunks)
    _cleanup_old_chroma(keep_hash=paths_hash)
    return retriever, all_chunks, total_pages, doc_info, vectorstore


def rebuild_retriever(pdf_paths: list):
    get_retriever.clear()
    h = _paths_hash(tuple(pdf_paths))
    return get_retriever(h, tuple(pdf_paths))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="centered")
st.title("📄 RAG Chatbot")

if not ANTHROPIC_API_KEY:
    st.error("Nastav `ANTHROPIC_API_KEY` v `.env` nebo `.streamlit/secrets.toml`.")
    st.stop()

# Session state
if "pdf_paths" not in st.session_state:
    persisted = sorted(UPLOAD_DIR.glob("*.pdf"))
    st.session_state.pdf_paths = [str(p) for p in persisted]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "evaluated" not in st.session_state:
    st.session_state.evaluated = set()

if st.session_state.pdf_paths:
    current_hash = _paths_hash(tuple(st.session_state.pdf_paths))
    retriever, chunks, num_pages, doc_info, vectorstore = get_retriever(current_hash, tuple(st.session_state.pdf_paths))
else:
    retriever, chunks, num_pages, doc_info, vectorstore = None, [], 0, [], None

# Vygeneruj uvítací zprávu pokud je chat prázdný a dokumenty jsou načteny
if not st.session_state.messages and doc_info:
    with st.spinner("Analyzuji dokumenty…"):
        welcome = build_welcome_message(doc_info, chunks)
    st.session_state.messages = [{
        "role": "assistant",
        "content": welcome,
        "pages": [],
        "question": "",
        "source_files": "",
        "latency_s": "",
    }]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dokument")

    uploaded_files = st.file_uploader("📤 Nahrát PDF", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        added = []
        for uf in uploaded_files:
            dest = UPLOAD_DIR / uf.name
            dest.write_bytes(uf.getbuffer())
            if str(dest) not in st.session_state.pdf_paths:
                st.session_state.pdf_paths.append(str(dest))
                added.append(uf.name)
        if added:
            retriever, chunks, num_pages, doc_info, vectorstore = rebuild_retriever(st.session_state.pdf_paths)
            st.session_state.messages = []
            st.session_state.evaluated = set()
            st.success(f"✅ Přidáno: {', '.join(added)}")
            st.rerun()

    st.write("**Načtené soubory:**")
    for path_str in list(st.session_state.pdf_paths):
        p = Path(path_str)
        col_name, col_btn = st.columns([5, 1])
        col_name.write(p.name)
        if col_btn.button("✕", key=f"remove_{p.name}"):
            st.session_state.pdf_paths.remove(path_str)
            if p.parent == UPLOAD_DIR and p.exists():
                p.unlink()
            if st.session_state.pdf_paths:
                retriever, chunks, num_pages, doc_info, vectorstore = rebuild_retriever(st.session_state.pdf_paths)
            else:
                get_retriever.clear()
            st.session_state.messages = []
            st.session_state.evaluated = set()
            st.rerun()

    st.write(f"**Celkem stránek:** {num_pages}")
    st.write(f"**Celkem chunků:** {len(chunks)}")

    if st.button("🔄 Obnovit přehled"):
        st.session_state.messages = []
        st.session_state.evaluated = set()
        st.rerun()

    st.divider()

    st.header("Evaluace")
    total, pct_positive, last20_ratings = load_eval_stats()
    st.metric("Celkem hodnocení", total)
    st.metric("Pozitivních odpovědí", f"{pct_positive} %" if total > 0 else "—")

    if last20_ratings:
        import pandas as pd
        pos = last20_ratings.count("👍")
        neg = last20_ratings.count("👎")
        df_chart = pd.DataFrame({"Počet": [pos, neg]}, index=["👍 Pozitivní", "👎 Negativní"])
        st.bar_chart(df_chart)

    col1, col2 = st.columns(2)
    with col1:
        if EVAL_CSV.exists():
            st.download_button(
                label="Stáhnout CSV",
                data=EVAL_CSV.read_bytes(),
                file_name="evaluations.csv",
                mime="text/csv",
            )
    with col2:
        if st.button("🗑️ Smazat"):
            clear_evaluations()
            st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "confidence" in msg and msg.get("question"):
                show_confidence_badge(msg["confidence"])
            if msg.get("pages"):
                st.caption(f"Zdroje — stránky PDF: {msg['pages']}")
            if msg.get("latency_s"):
                st.caption(f"Odpověď za {msg['latency_s']} s")

if not st.session_state.pdf_paths:
    st.info("Nahraj PDF dokument v sidebaru.")
    st.stop()

if question := st.chat_input("Napiš otázku k dokumentu…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Hledám odpověď…"):
            t0 = time.time()
            answer, source_docs = invoke_answer(retriever, question)
            latency_s = round(time.time() - t0, 1)
            confidence = compute_confidence(vectorstore, question) if vectorstore else 0.0

        st.markdown(answer)
        show_confidence_badge(confidence)
        pages = sorted({doc.metadata.get("page", "?") + 1 for doc in source_docs}) if source_docs else []
        if pages:
            st.caption(f"Zdroje — stránky PDF: {pages}")
        st.caption(f"Odpověď za {latency_s} s")

    source_files = ", ".join(sorted({
        Path(doc.metadata["source"]).name
        for doc in source_docs
        if doc.metadata.get("source")
    }))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages,
        "question": question,
        "source_files": source_files,
        "latency_s": latency_s,
        "confidence": confidence,
    })

# ---------------------------------------------------------------------------
# Evaluační vrstva
# ---------------------------------------------------------------------------
assistant_msgs = [(i, m) for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]

if assistant_msgs:
    last_idx, last_msg = assistant_msgs[-1]
    if last_idx not in st.session_state.evaluated:
        with st.container():
            st.divider()
            st.write("**Jak hodnotíš tuto odpověď?**")
            col1, col2 = st.columns(2)
            thumbs_up = col1.button("👍 Dobrá odpověď", key=f"up_{last_idx}")
            thumbs_down = col2.button("👎 Špatná odpověď", key=f"down_{last_idx}")
            comment = st.text_input(
                "Volitelný komentář",
                placeholder="Napiš komentář k odpovědi…",
                key=f"comment_{last_idx}",
            )
            if thumbs_up or thumbs_down:
                save_evaluation(
                    question=last_msg.get("question", ""),
                    answer=last_msg["content"],
                    pages=last_msg.get("pages", []),
                    rating="👍" if thumbs_up else "👎",
                    comment=comment,
                    document=last_msg.get("source_files", ""),
                    latency_s=last_msg.get("latency_s", ""),
                    confidence=last_msg.get("confidence", ""),
                )
                st.session_state.evaluated.add(last_idx)
                st.rerun()
