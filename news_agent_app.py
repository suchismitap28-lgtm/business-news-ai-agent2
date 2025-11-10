import os
from typing import List, Dict, Optional
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Optional content extraction
try:
    import trafilatura
except Exception:
    trafilatura = None

# OpenAI and Hugging Face setup
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

HF_QA_MODEL = os.environ.get("HF_QA_MODEL", "google/flan-t5-base")  # faster model
_hf_qa = None


def _init_hf():
    """Initialize Hugging Face fallback model."""
    global _hf_qa
    if _hf_qa is None:
        from transformers import pipeline
        _hf_qa = pipeline("text2text-generation", model=HF_QA_MODEL)


def _safe_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    """Safely get webpage content with headers."""
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
            print(f"⚠️ Bing returned {resp.status_code} for {url}")
            return None
        return resp
    except Exception as e:
        print(f"⚠️ Request error: {e}")
        return None


def search_news_bing(topic: str, max_links: int = 5):
    """Scrape Bing News for topic and return recent links."""
    from urllib.parse import quote
    q = quote(topic)
    url = f"https://www.bing.com/news/search?q={q}&qft=sortbydate%3d%221%22"
    st.write(f"🔍 Searching Bing for: {topic}")

    resp = _safe_get(url)
    if resp is None:
        st.error("❌ Could not fetch news from Bing. Possibly blocked or offline.")
        return []

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        st.error(f"⚠️ HTML parsing failed: {e}")
        return []

    results = []
    for a in soup.select("a.title, a[href*='http']"):
        href = a.get("href", "")
        title = (a.get_text(' ', strip=True) or "").strip()
        if not href or not title:
            continue
        if href.startswith('/') or "bing.com" in href:
            continue
        results.append({"title": title, "url": href, "source": "Bing News"})
        if len(results) >= max_links:
            break

    # Deduplicate
    dedup, seen = [], set()
    for r in results:
        u = r["url"].split("#")[0]
        if u not in seen:
            seen.add(u)
            dedup.append(r)

    if not dedup:
        st.warning("⚠️ No relevant articles found. Try another keyword (e.g., 'Lenskart IPO').")
    return dedup[:max_links]


def extract_article(url: str):
    """Extract clean article text and title."""
    resp = _safe_get(url, timeout=20)
    if not resp:
        return {"url": url, "title": "", "text": ""}
    text = ""
    title = ""
    if trafilatura:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
        except Exception:
            pass
    if not text:
        soup = BeautifulSoup(resp.text, "html.parser")
        node = (soup.select_one("article") or soup.select_one("[role='article']")
                or soup.select_one(".article") or soup.select_one(".post") or soup.select_one(".story"))
        if node:
            text = node.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    if not title:
        soup2 = BeautifulSoup(resp.text, "html.parser")
        if soup2.title and soup2.title.string:
            title = soup2.title.string.strip()
    return {"url": url, "title": title or "", "text": text or ""}


def has_openai():
    key = os.environ.get("OPENAI_API_KEY")
    return bool(key and _openai_mode is not None)


def openai_chat(messages, model=None, temperature=0.3, max_tokens=600):
    """Handle OpenAI Chat API (new or legacy)."""
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if _openai_mode == "new":
        try:
            client = OpenAI()
            r = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception:
            pass
    try:
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        r = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def hf_answer(question: str, context: str):
    """Fallback: Answer using Hugging Face model."""
    _init_hf()
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return _hf_qa(prompt, max_new_tokens=256)[0]["generated_text"].strip()


def compress_text(text: str, words: int = 200):
    """Shorten long article text."""
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:120] + ["..."] + parts[-60:])


# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="Business News AI Agent", page_icon="🗞️", layout="wide")
st.title("🗞️ Business News AI Agent (Fast Q&A Mode)")

with st.sidebar:
    st.markdown("### Settings")
    topic = st.text_input("Topic", value="Lenskart IPO 2025")
    max_links = st.slider("Max article links", min_value=2, max_value=10, value=4, step=1)
    question = st.text_area("Your question", value="What are the major investor risks mentioned in recent news?")
    use_run = st.button("Fetch & Answer")

if use_run:
    with st.spinner("🔎 Searching latest news..."):
        links = search_news_bing(topic, max_links=max_links)

    st.success(f"Found {len(links)} articles.")
    articles = []
    prog = st.progress(0.0)

    for i, r in enumerate(links, 1):
        a = extract_article(r["url"])
        a["title"] = a.get("title") or r.get("title") or ""
        articles.append(a)
        prog.progress(i / len(links))

    st.subheader("📰 Sources")
    for a in articles:
        st.markdown(f"- [{a.get('title') or 'Untitled'}]({a.get('url')})")

    # Build concise context
    context_blocks = []
    for i, a in enumerate(articles[:5], 1):
        text = a.get("text", "")
        if not text or len(text.split()) < 50:
            continue
        src = f"[{a.get('title') or 'Untitled'}]({a.get('url')})"
        context_blocks.append(f"({i}) SOURCE: {src}\n\n{compress_text(text)}")

    context = "\n\n---\n\n".join(context_blocks)

    if not context_blocks:
        st.error("❌ No valid news content found for your topic. Try again later.")
    else:
        st.info(f"🧠 Using {len(context_blocks)} sources for analysis.")

        with st.spinner("🧩 Generating AI-based answer..."):
            messages = [
                {"role": "system", "content": (
                    "You are a senior business analyst. "
                    "Answer the user's question concisely using only the provided sources. "
                    "Always include clickable markdown links to cite sources."
                )},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
            ]
            try:
                ans = openai_chat(messages, max_tokens=600)
            except Exception:
                ans = hf_answer(question, context)

        st.markdown("### 💡 Answer")
        st.markdown(ans)
        st.success("✅ Done")
