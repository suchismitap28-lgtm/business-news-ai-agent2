
import os
from typing import List, Dict, Optional
import streamlit as st
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:
    trafilatura = None

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

HF_SUM_MODEL = os.environ.get("HF_SUM_MODEL", "facebook/bart-large-cnn")
HF_QA_MODEL  = os.environ.get("HF_QA_MODEL", "google/flan-t5-large")
_hf_sum = None
_hf_qa  = None

def _init_hf():
    global _hf_sum, _hf_qa
    if _hf_sum is None:
        from transformers import pipeline
        _hf_sum = pipeline("summarization", model=HF_SUM_MODEL)
    if _hf_qa is None:
        from transformers import pipeline
        _hf_qa = pipeline("text2text-generation", model=HF_QA_MODEL)

def _safe_get(url: str, headers: Optional[Dict[str,str]] = None, timeout: int = 15):
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
        resp.raise_for_status()
        return resp
    except Exception:
        return None

def search_news_bing(topic: str, max_results: int = 10):
    from urllib.parse import quote
    import streamlit as st
    q = quote(topic)
    url = f"https://www.bing.com/news/search?q={q}&qft=sortbydate%3d%221%22"
    
    # Fetch page
    resp = _safe_get(url)
    if resp is None:
        st.error("❌ Could not fetch news from Bing. Check internet or try again later.")
        return []
    
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        st.error(f"⚠️ Error parsing Bing News results: {e}")
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
        if len(results) >= max_results:
            break

    # Deduplicate
    seen, dedup = set(), []
    for r in results:
        u = r["url"].split("#")[0]
        if u not in seen:
            seen.add(u)
            dedup.append(r)

    if not dedup:
        st.warning("⚠️ No recent articles found. Try another topic like 'Lenskart IPO' or 'EV policy India'.")
    return dedup[:max_results]

def extract_article(url: str):
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

def chunk_text(text: str, max_tokens: int = 1200):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)] if words else []

def has_openai():
    key = os.environ.get("OPENAI_API_KEY")
    return bool(key and _openai_mode is not None)

def openai_chat(messages, model=None, temperature=0.3, max_tokens=700):
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if _openai_mode == "new":
        try:
            from openai import OpenAI
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

CONSULTANT_TONE = (
    "Use a concise, consultant-style tone. Start with a one-line takeaway, "
    "then 3-5 bullets with implications/risks/opportunities, then an optional "
    "'So what' for operators/investors/policy.'"
)

def summarize_text(text: str):
    chunks = chunk_text(text)
    summaries = []
    if has_openai():
        for ch in chunks:
            messages = [
                {"role": "system", "content": f"You are a business analyst. {CONSULTANT_TONE}"},
                {"role": "user", "content": f"Summarize this news article segment in 5-7 concise sentences:\\n\\n{ch}"}
            ]
            try:
                summaries.append(openai_chat(messages, max_tokens=350))
            except Exception:
                break
    if not summaries:
        _init_hf()
        res = []
        for ch in chunks:
            res.append(_hf_sum(ch, max_length=220, min_length=60, do_sample=False)[0]["summary_text"])
        summaries = [r.strip() for r in res]
    return "\\n\\n".join(summaries)

def answer_question(question: str, articles: List[Dict[str,str]]):
    arts = sorted([a for a in articles if a.get("text")], key=lambda a: len(a["text"]), reverse=True)[:5]
    blocks = []
    for a in arts:
        src = f"[{a.get('title') or 'Untitled'}]({a.get('url')})"
        ctx = a.get("summary") or a.get("text") or ""
        ctx = " ".join(ctx.split()[:450])
        blocks.append(f"SOURCE: {src}\\nCONTEXT: {ctx}")
    context = "\\n\\n---\\n\\n".join(blocks) if blocks else "No context available."
    if has_openai():
        messages = [
            {"role": "system", "content": f"You are a senior strategy consultant. {CONSULTANT_TONE} Cite sources inline as [n]."},
            {"role": "user", "content": f"Use the sources and context below to answer the question. If something is not supported by the sources, say you don't know.\\n\\n{context}\\n\\nQuestion: {question}"}
        ]
        try:
            return openai_chat(messages, max_tokens=700)
        except Exception:
            pass
    _init_hf()
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the answer isn't in the context, say you don't know.\\n\\n"
        f"Context:\\n{context}\\n\\nQuestion: {question}\\nAnswer:"
    )
    return _hf_qa(prompt, max_new_tokens=256)[0]["generated_text"].strip()

st.set_page_config(page_title="Business News AI Agent", page_icon="🗞️", layout="wide")
st.title("🗞️ Business News AI Agent (Hybrid)")

with st.sidebar:
    st.markdown("### Settings")
    topic = st.text_input("Topic", value="electric vehicle battery recycling policy India")
    max_links = st.slider("Max article links", min_value=3, max_value=20, value=8, step=1)
    question = st.text_area("Your question", value="What are the key business implications over the next 12 months?")
    use_run = st.button("Fetch & Analyze")

if use_run:
    with st.spinner("Searching latest news..."):
        links = search_news_bing(topic, max_links=max_links)
    st.success(f"Found {len(links)} links.")
    articles = []
    prog = st.progress(0.0)
    for i, r in enumerate(links, 1):
        a = extract_article(r["url"])
        a["title"] = a.get("title") or r.get("title") or ""
        articles.append(a)
        prog.progress(i/len(links))
    st.subheader("Articles")
    for a in articles:
        st.markdown(f"- [{a.get('title') or 'Untitled'}]({a.get('url')})")
    st.subheader("Summaries")
    enriched = []
    for a in articles:
        txt = a.get("text") or ""
        if len(txt.split()) < 80:
            continue
        with st.spinner(f"Summarizing: {a.get('title')[:60]}..."):
            try:
                s = summarize_text(txt)
            except Exception:
                s = ""
        a["summary"] = s
        if s:
            st.markdown(f"**{a.get('title')}**")
            st.write(s)
        enriched.append(a)
    st.subheader("Q&A")
    with st.spinner("Answering your question..."):
        ans = answer_question(question, enriched or articles)
    st.markdown(ans)
