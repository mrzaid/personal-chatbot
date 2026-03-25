import subprocess
from pathlib import Path
from typing import List
import streamlit as st
from qa_prompts import PROMPT_TMPL

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient

import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
HF_API_TOKEN = (os.getenv("HUGGING_FACE_API_TOKEN") or "").strip()

EMBED_MODEL_NAME = os.getenv("HUGGING_FACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-flash")

INDEX_DIR = ROOT_DIR / "data_index"

CONVERSATION_STARTERS = [
    "Who is Zaid Ahmed?",
    "Summarize your work experience.",
    "What are your strongest technical skills?",
    "Tell me about your education and certifications.",
    "How would you fit a senior engineering role?",
]


class HFAPIEmbeddings(Embeddings):
    def __init__(self, repo_id: str, token: str | None = None, timeout: float = 120.0):
        self.client = InferenceClient(model=repo_id, token=token, timeout=timeout)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.feature_extraction(texts)

    def embed_query(self, text: str) -> List[float]:
        vec = self.client.feature_extraction(text)
        return vec[0] if (isinstance(vec, list) and vec and isinstance(vec[0], list)) else vec


def build_chain_gemini(retriever, _llm_repo, _max_new, _temp, _show_sources):
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in your .env to use the Gemini inference endpoint.")

    llm = ChatGoogleGenerativeAI(
        model=_llm_repo,
        api_key=GOOGLE_API_KEY,
        temperature=_temp,
        max_output_tokens=_max_new,
        convert_system_message_to_human=True,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TMPL,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=_show_sources,
    )
    return qa


# ========================= Streamlit UI =========================
st.set_page_config(
    page_title="Zaid Ahmed · Profile Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 880px; }
        h1.hero-title {
            font-size: 1.75rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.03em;
            color: #0f172a !important;
            margin-bottom: 0.15rem !important;
        }
        p.hero-sub {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 0;
        }
        div[data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        .starter-wrap { margin-top: 0.35rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if not INDEX_DIR.exists():
    with st.spinner("Index not found. Building FAISS index (first run)…"):
        proc = subprocess.run(["python", "src/ingest.py"], capture_output=True, text=True)
        if proc.returncode != 0:
            st.error(f"ingest.py failed:\n{proc.stderr}")
            st.stop()

hf_token = HF_API_TOKEN

st.sidebar.markdown("### Settings")
if not hf_token:
    st.sidebar.warning("HUGGING_FACE_API_TOKEN is not set in `src/.env` (optional for local embeddings).")
if not GOOGLE_API_KEY:
    st.sidebar.error(
        "Set GOOGLE_API_KEY in src/.env using a key from https://aistudio.google.com/app/apikey — then restart the app."
    )

show_sources = st.sidebar.toggle("Show source excerpts", value=False)
st.sidebar.divider()
st.sidebar.caption("Models")
st.sidebar.code(EMBED_MODEL_NAME, language=None)
st.sidebar.caption("Embedding model")
st.sidebar.code(LLM_MODEL_NAME, language=None)
st.sidebar.caption("Chat model")

k = 4
max_new_tokens = 512
temperature = 0.1


@st.cache_resource(show_spinner=True)
def _load_chain(_store_dir: str, _embed_repo: str, _llm_repo: str, _k: int, _max_new: int, _temp: float, _show_sources: bool):
    if not Path(_store_dir).exists():
        raise FileNotFoundError(f"FAISS store not found at '{_store_dir}'. Run ingest.py first.")
    embeddings = HFAPIEmbeddings(repo_id=_embed_repo, token=hf_token)
    vs = FAISS.load_local(
        _store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vs.as_retriever(search_kwargs={"k": _k})
    return build_chain_gemini(retriever, _llm_repo, _max_new, _temp, _show_sources)


with st.spinner("Preparing retriever & LLM…"):
    chain = _load_chain(INDEX_DIR, EMBED_MODEL_NAME, LLM_MODEL_NAME, k, max_new_tokens, temperature, show_sources)

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(docs):
    if not docs:
        return
    st.markdown("**Sources**")
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        label = f"{Path(src).name}" + (f" (page {page + 1})" if isinstance(page, int) else "")
        with st.expander(f"{i}. {label}"):
            st.write(d.page_content[:1500] + ("…" if len(d.page_content) > 1500 else ""))


def _friendly_api_error(err: str) -> str:
    if "API_KEY_INVALID" in err or "API key not valid" in err:
        return (
            "Gemini rejected your API key (`API_KEY_INVALID`). "
            "Create a new key at https://aistudio.google.com/app/apikey , put it in `src/.env` as "
            "`GOOGLE_API_KEY=...` (no quotes, no spaces around `=`), save, and restart Streamlit."
        )
    return err


# Process queued message (chat submit or starter) before rendering
pending = st.session_state.pop("_turn_query", None)
if pending:
    pending = pending.strip()
    if pending:
        try:
            with st.spinner("Thinking…"):
                res = chain.invoke({"query": pending})
                if isinstance(res, dict):
                    answer = res.get("result", "")
                    sources = res.get("source_documents", []) if show_sources else []
                else:
                    answer, sources = str(res), []
        except Exception as e:
            err = _friendly_api_error(str(e))
            answer, sources = f"[error] {err}", []
        st.session_state.history.append((pending, answer, sources))

_, center, _ = st.columns([1, 2.2, 1])
with center:
    st.markdown('<p class="hero-title">Zaid Ahmed</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Profile assistant · Grounded answers from resume & profile documents</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    for q, a, srcs in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if show_sources:
                render_sources(srcs)

    if not st.session_state.history:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi — I answer questions about **Zaid Ahmed** for recruiters and hiring teams. "
                "Use a starter below or type your own question."
            )
            st.markdown('<div class="starter-wrap"></div>', unsafe_allow_html=True)
            for i in range(0, len(CONVERSATION_STARTERS), 2):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        CONVERSATION_STARTERS[i],
                        key=f"starter_{i}",
                        use_container_width=True,
                    ):
                        st.session_state._turn_query = CONVERSATION_STARTERS[i]
                        st.rerun()
                if i + 1 < len(CONVERSATION_STARTERS):
                    with c2:
                        if st.button(
                            CONVERSATION_STARTERS[i + 1],
                            key=f"starter_{i + 1}",
                            use_container_width=True,
                        ):
                            st.session_state._turn_query = CONVERSATION_STARTERS[i + 1]
                            st.rerun()

    if prompt := st.chat_input("Ask about experience, skills, education, or role fit…"):
        st.session_state._turn_query = prompt.strip()
        st.rerun()
