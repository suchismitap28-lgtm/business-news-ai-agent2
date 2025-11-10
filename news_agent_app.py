import os
from typing import List, Dict, Optional
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Optional article extractor
try:
    import trafilatura
except Exception:
    trafilatura = None

# --- OpenAI & Hugging Face setup ---
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


def _safe_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    """Safe request wrapper"""
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    if headers:
        default_headers.update(headers)
    try:
        resp = requests.get(url, headers=default_headers, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None
        return resp
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def search_news_bing(topic: str, max_links: int = 5):
    """Fetch recent news links from Bing."""
    from urllib.parse import quote
    q = quote(topic)
    url = f"https://www.bing.com/news/search?q={q}&qft=sortbydate%3d%221%22"
    resp = _safe_get(url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a.title, a[href*='http']"):
        href = a.get("href", "")
        title = (a.get_text(' ', strip=True) or "").strip()
        if not href or not title:
            continue
        if href.startswith('/') or "bing.com" in href:
            continue
        results.append({"title": title, "url": href})
        if len(results) >= max_links:
            break
    seen, dedup = set(), []
    for r in results:
        u = r["url"].split("#")[0]
        if u not in seen:
            seen.add(u)
            dedup.append(r)
    return dedup[:max_links]


@st.cache_data(show_spinner=False)
def extract_article(url: str):
    """Extract readable text from news."""
    resp = _safe_get(url, timeout=20)
    if not resp:
        return {"url": url, "title": "", "text": ""}
    text, title = "", ""
    if trafilatura:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
        except Exception:
            pass
    if not text:
        soup = BeautifulSoup(resp.text, "html.parser")
        node = soup.select_one("article") or soup.select_one(".post") or soup.select_one("[role='article']")
        if node:
            text = node.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    return {"url": url, "title": title or "", "text": text or ""}


def has_openai():
    return bool(os.environ.get("OPENAI_API_KEY") and _openai_mode is not None)


def openai_chat(messages, model=None, temperature=0.4, max_tokens=1500):
    """Handle OpenAI calls gracefully."""
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if _openai_mode == "new":
        client = OpenAI()
        r = client.chat.completions.create(model=model, messages=messages,
                                           temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    r = openai.ChatCompletion.create(model=model, messages=messages,
                                     temperature=temperature, max_tokens=max_tokens)
    return r["choices"][0]["message"]["content"].strip()


def hf_answer(question: str, context: str):
    """Fallback using HF model."""
    _init_hf()
    prompt = (
        "Answer each question separately in a structured, professional format. "
        "Start each with 'Q1', 'Q2', etc. followed by an explanation and sources.\n\n"
        f"Context:\n{context}\n\nQuestions:\n{question}\nAnswers:"
    )
    return _hf_qa(prompt, max_new_tokens=700)[0]["generated_text"].strip()


def compress_text(text: str, words: int = 220):
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:140] + ["..."] + parts[-80:])


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Business News AI Agent", page_icon="🗞️", layout="wide")

# Polished minimal CSS
st.markdown("""
<style>
h3, h4 {
    color: #003366;
    font-weight: 700;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}
div.answer-block {
    background-color: #f8fafc;
    border-left: 4px solid #0073e6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 1.2em;
    line-height: 1.5;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.06);
    font-size: 14px;
    font-weight: 400;
    color: #1a1a1a;
}
b {
    color: #004080;
    font-weight: 600;
}
a {
    color: #0055cc;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🗞️ Business News AI Agent — Structured Insights")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    topic = st.text_input("Topic", value="Lenskart IPO 2025")
    max_links = st.slider("Max article links", 2, 10, 5)

    # Dynamic questions
    if "questions" not in st.session_state:
        st.session_state["questions"] = []

    new_q = st.text_input("Enter a question:")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Question"):
            if new_q.strip():
                st.session_state["questions"].append(new_q.strip())
    with col2:
        if st.button("🗑️ Clear All"):
            st.session_state["questions"] = []

    if st.session_state["questions"]:
        st.markdown("#### 📋 Your Questions")
        for i, q in enumerate(st.session_state["questions"], 1):
            st.markdown(f"{i}. {q}")

    run_btn = st.button("🧠 Generate Answers")

if run_btn:
    if not st.session_state["questions"]:
        st.error("❌ Please add at least one question.")
        st.stop()

    questions_text = "\n".join([f"Q{i}. {q}" for i, q in enumerate(st.session_state['questions'], 1)])

    with st.spinner("Fetching latest news..."):
        links = search_news_bing(topic, max_links=max_links)

    if not links:
        st.error("❌ No relevant news found.")
        st.stop()

    st.success(f"✅ Found {len(links)} relevant sources.")
    articles = []
    prog = st.progress(0.0)
    for i, r in enumerate(links, 1):
        a = extract_article(r["url"])
        a["title"] = a.get("title") or r.get("title") or ""
        articles.append(a)
        prog.progress(i / len(links))

    st.subheader("📰 Sources Used")
    for a in articles:
        st.markdown(f"- [{a.get('title') or 'Untitled'}]({a.get('url')})")

    # Context building
    context_blocks = []
    for i, a in enumerate(articles[:5], 1):
        text = a.get("text", "")
        if not text or len(text.split()) < 50:
            continue
        src = f"[{a.get('title') or 'Untitled'}]({a.get('url')})"
        context_blocks.append(f"({i}) SOURCE: {src}\n\n{compress_text(text)}")
    context = "\n\n---\n\n".join(context_blocks)

    st.info(f"🧠 Using {len(context_blocks)} sources for analysis...")

    with st.spinner("Generating descriptive, structured answers..."):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior business analyst. "
                    "Answer each question as a separate section formatted as:\n\n"
                    "### Q1. <Question>\n"
                    "<div class='answer-block'>\n"
                    "<b>Summary:</b> one insightful line.<br>\n"
                    "Then 4–6 lines of descriptive analysis using data, trends, and insights.<br>\n"
                    "End with **Sources:** and markdown links.\n"
                    "</div>\n"
                    "Use clear paragraphs. Avoid emojis or decorative numbering."
                )
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\nQuestions:\n{questions_text}\n\nContext:\n{context}"
            }
        ]

        try:
            ans = openai_chat(messages, max_tokens=1500)
        except Exception:
            ans = hf_answer(questions_text, context)

    # --- Final clean formatting ---
    ans = ans.replace("</div>", "</div><br>")
    st.markdown("### 💡 Analytical Answers")
    st.markdown(ans, unsafe_allow_html=True)
    st.success("✅ Done — neatly formatted insights generated!")
