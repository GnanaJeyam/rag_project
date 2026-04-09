import asyncio
import logging
import os
import threading
import time
import traceback
from pathlib import Path

import google.genai as geni
import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

from pdf_util import save_pdf

load_dotenv()
logger = logging.getLogger("uvicorn")
client = geni.Client()
INNGEEST_API_URL = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

@st.cache_resource
def _get_bg_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _get_bg_loop())
    return future.result(timeout=60)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_testing_app", logger=logger, is_production=False)

async def send_rag_ingest_event(pdf_path: Path) -> None:
    await get_inngest_client().send(
        inngest.Event(
            name="rag/pdf_uploaded",
            data={
                'file_path': str(pdf_path.resolve())
            }
        )
    )

st.set_page_config(page_title="RAG — PDF Q&A", page_icon="📄", layout="centered")

# ── Custom CSS for a classic look ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMainMenu"], [data-testid="stToolbar"] { display: none; }
    .block-container { max-width: 720px; padding-top: 2rem; }
    h1 { letter-spacing: -0.5px; }
    .stDivider { margin: 1.5rem 0; }
    /* prevent any text from overflowing containers */
    code, pre, .stCaption, .stMarkdown, .stExpander,
    [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📄 RAG — PDF Question & Answer")
st.caption("Upload documents, then ask questions — answers are grounded in the actual text.")

# ── PDF Upload Section ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("1 · Upload a PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if st.session_state.get("processed_file") != uploaded.name:
            file_path = save_pdf(uploaded)
            try:
                with st.spinner("Ingesting — parsing, chunking & embedding …"):
                    run_async(send_rag_ingest_event(file_path))
                    time.sleep(1)
                    st.session_state["processed_file"] = uploaded.name
                st.success(f"✅  **{uploaded.name}** ingested successfully!")
            except Exception as e:
                st.error(f"❌ Upload failed: **{type(e).__name__}** — {e}")
                st.code(traceback.format_exc(), language="text")

st.divider()

# ── Helper: poll Inngest for run output ────────────────────────────────────────
def get_inngest_run_details(run_id: str) -> dict:
    start_time = time.time()
    timeout_s: float = 60.0
    poll_interval_s: float = 0.5

    while True:
        url = f"{INNGEEST_API_URL}/events/{run_id}/runs"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("data", [])
        if data:
            run = data[0]
            status = run.get("status")
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest run {status}")
        if time.time() - start_time > timeout_s:
            raise TimeoutError("Inngest run timed out")
        time.sleep(poll_interval_s)


async def send_query_event(qstn: str) -> str:
    result = await get_inngest_client().send(
        inngest.Event(
            name="rag/query_data",
            data={"question": qstn},
        )
    )
    return result[0]


# ── Q&A Section ────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("2 · Ask a Question")

    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            label="Your question",
            placeholder="e.g.  What are the key skills mentioned in the resume?",
            height=120,
            label_visibility="collapsed",
        )

        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit = st.form_submit_button("🔍  Submit", use_container_width=True, type="primary")

    # Clear button lives outside the form
    if st.button("🗑  Clear", use_container_width=False):
        for key in ("query_answer", "query_sources"):
            st.session_state.pop(key, None)
        st.rerun()

    if submit:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Thinking — embedding query & searching …"):
                    event_id = run_async(send_query_event(question))
                    output = get_inngest_run_details(event_id)
                    st.session_state["query_answer"] = output.get("answer", "No answer")
                    st.session_state["query_sources"] = set(output.get("sources", []))
            except Exception as e:
                st.error(f"❌ Query failed: **{type(e).__name__}** — {e}")
                st.code(traceback.format_exc(), language="text")

# ── Answer Display ─────────────────────────────────────────────────────────────
if "query_answer" in st.session_state:
    st.divider()

    with st.container(border=True):
        st.subheader("💡 Answer")
        st.markdown(st.session_state["query_answer"] or "_No answer available._")

    sources = st.session_state.get("query_sources", [])
    if sources:
        with st.expander("📎  Sources", expanded=True):
            for src in sources:
                name = Path(src).name
                st.markdown(f"- 📄 **{name}**")
                st.caption(f"  `{src}`")
