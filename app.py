"""
AI Knowledge Assistant
Ask questions about your own documents (.txt / .md).

Run:  streamlit run app.py
Optional: set ANTHROPIC_API_KEY to get natural-language answers.
"""
import os
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = Path("knowledge")
CHUNK_WORDS = 120
OVERLAP = 20
TOP_K = 3
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


# ---------- 1. Load and chunk documents ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


def load_documents():
    chunks = []  # list of (source_name, chunk_text)
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    for path in KNOWLEDGE_DIR.glob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for c in chunk_text(text):
                if c.strip():
                    chunks.append((path.name, c))
    return chunks


# ---------- 2. Build search index ----------
@st.cache_resource
def build_index(_version):
    chunks = load_documents()
    if not chunks:
        return chunks, None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c for _, c in chunks])
    return chunks, vectorizer, matrix


def search(query, chunks, vectorizer, matrix, k=TOP_K):
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    best = scores.argsort()[::-1][:k]
    return [(chunks[i][0], chunks[i][1], float(scores[i])) for i in best if scores[i] > 0]


# ---------- 3. Generate answer ----------
def generate_answer(question, results):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    context = "\n\n".join(f"[{src}] {text}" for src, text, _ in results)
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="Answer the question using ONLY the provided context. "
               "If the answer is not in the context, say you don't know.",
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text


# ---------- 4. UI ----------
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠")
st.title("🧠 AI Knowledge Assistant")
st.caption("Ask questions about your documents.")

with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add .txt / .md files", type=["txt", "md"],
                               accept_multiple_files=True)
    if uploads:
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (KNOWLEDGE_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploads)} file(s).")
    files = sorted(p.name for p in KNOWLEDGE_DIR.glob("*") if p.suffix in {".txt", ".md"})
    st.write(files or "No documents yet.")
    st.write("Mode:", "🤖 AI answers" if os.getenv("ANTHROPIC_API_KEY") else "🔎 Search only")

version = tuple((p.name, p.stat().st_mtime) for p in KNOWLEDGE_DIR.glob("*")) if KNOWLEDGE_DIR.exists() else ()
chunks, vectorizer, matrix = build_index(version)

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

question = st.chat_input("Ask a question...")
if question:
    st.chat_message("user").markdown(question)
    st.session_state.history.append(("user", question))

    if matrix is None:
        answer = "Please add some documents first."
    else:
        results = search(question, chunks, vectorizer, matrix)
        if not results:
            answer = "I couldn't find anything relevant in the documents."
        else:
            ai = generate_answer(question, results)
            sources = "\n".join(f"- **{s}** (score {sc:.2f}): {t[:200]}..." for s, t, sc in results)
            answer = (ai + "\n\n---\n**Sources**\n" + sources) if ai else ("**Top matches**\n" + sources)

    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append(("assistant", answer))
