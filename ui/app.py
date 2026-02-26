"""
Streamlit UI — minimal, functional, deliberately not the focus.

Features:
  - File upload → POST /ingest
  - Chat interface → POST /query
  - Source citations displayed below each answer
  - Query rewrite indicator when the graph rewrote the question
"""
from __future__ import annotations

import requests
import streamlit as st
from requests import RequestException, Timeout

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Chat With Your Docs", page_icon="📄", layout="wide")
st.title("📄 Chat With Your Docs")


def _safe_request(method: str, url: str, **kwargs) -> requests.Response | None:
    try:
        return requests.request(method, url, **kwargs)
    except Timeout:
        st.error(
            "⏱️ Request timed out. The backend may be busy (embedding/ingest) or unavailable. "
            "Check the API logs, Qdrant status, and your OpenAI key."
        )
        return None
    except RequestException as exc:
        st.error(f"🚫 API request failed: {exc}")
        return None

# ── Sidebar: Upload ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    health = _safe_request("GET", f"{API_BASE}/health", timeout=3)
    if health and health.ok:
        data = health.json()
        status = data.get("status", "unknown")
        qdrant_status = data.get("qdrant", "unknown")
        llm_status = data.get("llm", "unknown")
        st.caption(f"API: {status} • Qdrant: {qdrant_status} • LLM: {llm_status}")
    else:
        st.caption("API: unavailable")
    uploaded = st.file_uploader(
        "PDF, TXT, MD, DOCX", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest Documents"):
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}..."):
                resp = _safe_request(
                    "POST",
                    f"{API_BASE}/ingest",
                    files={"file": (f.name, f.getvalue(), f.type)},
                    timeout=120,
                )
            if resp is None:
                continue
            if resp.status_code == 201:
                data = resp.json()
                st.success(f"✅ {data['filename']}: {data['chunks_indexed']} chunks indexed")
            else:
                st.error(f"❌ {f.name}: {resp.text}")

    st.divider()
    st.header("Indexed Documents")
    if st.button("Refresh list"):
        resp = _safe_request("GET", f"{API_BASE}/documents", timeout=10)
        if resp and resp.ok:
            for doc in resp.json().get("documents", []):
                st.write(f"📄 {doc['filename']}")

# ── Main: Chat ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    page = f", page {s['page']}" if s.get("page") else ""
                    st.caption(f"• {s['filename']}{page} (score: {s['score']:.3f})")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = _safe_request(
                "POST",
                f"{API_BASE}/query",
                json={"question": prompt},
                timeout=60,
            )
        if resp is None:
            err = "❌ Error: API request failed"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        elif resp.ok:
            data = resp.json()
            answer = data["answer"]

            if data.get("query_rewritten"):
                st.caption(f"🔄 Query rewritten to: *{data.get('rewritten_question')}*")

            st.markdown(answer)

            sources = data.get("sources", [])
            if sources:
                with st.expander("📎 Sources"):
                    for s in sources:
                        page = f", page {s['page']}" if s.get("page") else ""
                        st.caption(f"• {s['filename']}{page} (score: {s['score']:.3f})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            err = f"❌ Error: {resp.text}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
