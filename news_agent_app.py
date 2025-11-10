import os
from typing import List, Dict, Optional
import streamlit as st
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:
    trafilatura = None

# --- AI MODELS SETUP ---
_openai_mode = None
try:
    from openai import OpenAI
    _openai_mode = "new"
except Exception:
    try:
        import openai
        _openai_mode = "legacy"
    except Exception:
        _openai_mode = None


HF_QA_MODEL = os.environ.get("HF_QA_MODEL", "google/flan-t5-base")
_hf_qa = None


def _init_hf():
    global _hf_qa
    if _hf_qa is None:
        from transformers import pipeline
        _hf_qa = pipeline("text2text-generation", model=HF_QA_MODEL)


def _safe_get(url: str, timeout: int = 15):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def search_news_bing(topic: str, max_links: int = 5):
    """Scrape recent Bing News links."""
    from urllib.parse import quote
    q = quote(topic)
    url = f"https://www.bing.com/news/search?q={q}&qft=sortbydate%3d%221%22"
    resp = _safe_get(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("a.title, a[href*='http']"):
        href = a.get("href", "")
        title = (a.get_text(" ", strip=True) or "").strip()
        if href and title and not href.startswith("/") and "bing.com" not in href:
            links.append({"title": title, "url": href})
        if len(links) >= max_links:
            break
    return links


@st.cache_data(show_spinner=False)
def extract_article(url: str):
    """Extract readable article content."""
    resp = _safe_get(url, timeout=20)
    if not resp:
        return {"url": url, "title": "", "text": ""}
    text, title = "", ""
    if trafilatura:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded) or ""
        except Exception:
            pass
    if not text:
        soup = BeautifulSoup(resp.text, "html.parser")
        article = soup.select_one("article") or soup.select_one("p")
        if article:
            text = article.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
    return {"url": url, "title": title, "text": text}


def has_openai():
    return bool(os.environ.get("OPENAI_API_KEY") and _openai_mode is not None)


def openai_chat(messages, model="gpt-4o-mini", temperature=0.5, max_tokens=1000):
    """Call OpenAI model."""
    if _openai_mode == "new":
        client = OpenAI()
        res = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return res.choices[0].message.content.strip()
    else:
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        res = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return res["choices"][0]["message"]["content"].strip()


def hf_answer(question: str, context: str):
    """Fallback if no OpenAI key."""
    _init_hf()
    prompt = f"Answer the following question based only on the given context:\n\nContext:\n{context}\n\nQuestion: {question}"
    return _hf_qa(prompt, max_new_tokens=300)[0]["generated_text"].strip()


def compress_text(text: str, words: int = 250):
    parts = text.split()
    return " ".join(parts[:words]) if len(parts) > words else text


# -------------- STREAMLIT APP -----------------
st.set_page_config(page_title="Business News AI Agent", page_icon="🗞️", layout="wide")

st.markdown("""
<style>
div.answer-block {
    background-color: #f8fafc;
    border-left: 4px solid #0073e6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 1.2em;
    line-height: 1.6;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
    font-size: 14px;
    font-weight: 400;
    color: #222;
}
b {
    color: #004080;
}
a {
    color: #0055cc;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🗞️ Business News AI Agent — Insight Generator")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    topic = st.text_input("Topic", "Lenskart IPO 2025")
    max_links = st.slider("Max articles", 2, 10, 5)

    if "questions" not in st.session_state:
        st.session_state["questions"] = []

    new_q = st.text_input("Enter your question:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Question"):
            if new_q.strip():
                st.session_state["questions"].append(new_q.strip())
    with col2:
        if st.button("🗑️ Clear All"):
            st.session_state["questions"] = []

    if st.session_state["questions"]:
        st.markdown("#### Your Questions:")
        for i, q in enumerate(st.session_state["questions"], 1):
            st.markdown(f"{i}. {q}")

    run = st.button("🧠 Generate Answers")

if run:
    if not st.session_state["questions"]:
        st.error("❌ Please add at least one question.")
        st.stop()

    with st.spinner("Fetching latest news..."):
        links = search_news_bing(topic, max_links=max_links)

    if not links:
        st.error("No news found.")
        st.stop()

    st.success(f"Found {len(links)} sources.")
    articles = []
    for r in links:
        a = extract_article(r["url"])
        a["title"] = a.get("title") or r.get("title") or ""
        articles.append(a)

    context_blocks = []
    for a in articles:
        text = compress_text(a["text"])
        if len(text.split()) > 60:
            context_blocks.append(f"{a['title']}:\n{text}\n(Source: {a['url']})")
    context = "\n\n---\n\n".join(context_blocks)

    st.subheader("📰 Sources Used")
    for a in articles:
        st.markdown(f"- [{a['title']}]({a['url']})")

    st.markdown("### 💡 Analytical Answers")

    for i, question in enumerate(st.session_state["questions"], 1):
        with st.spinner(f"Answering Question {i}..."):
            messages = [
                {"role": "system", "content": (
                    "You are a senior business analyst. "
                    "Based on the provided news context, answer clearly and analytically. "
                    "Do NOT repeat the question text. Format as:\n"
                    "<div class='answer-block'>"
                    "<b>Summary:</b> One concise line.<br>"
                    "Then 4–6 descriptive lines with insights, context, and implications.<br>"
                    "End with **Sources:** markdown links."
                    "</div>"
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]

            try:
                answer = openai_chat(messages, max_tokens=800)
            except Exception:
                answer = hf_answer(question, context)

        st.markdown(f"#### Q{i}. {question}")
        st.markdown(answer, unsafe_allow_html=True)
