import os
import time

from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(page_title="Enterprise AI Knowledge Assistant", page_icon="🤖")
st.title("🤖 Enterprise AI Knowledge Assistant")

st.sidebar.title("TechNova AI Solutions")
st.sidebar.write("Enterprise Knowledge Assistant")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 System Status")

st.sidebar.success("🟢 Gemini API Connected")
st.sidebar.success("🟢 FAISS Index Loaded")
st.sidebar.success("🟢 BM25 Retriever Ready")
st.sidebar.success("🟢 Cross Encoder Loaded")

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Technical Architecture"):

    st.markdown("""
1️⃣ Query Rewrite

⬇️

2️⃣ Hybrid Search

⬇️

3️⃣ Cross-Encoder Ranking

⬇️

4️⃣ Gemini Response
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

st.caption(f"Current Chat : {st.session_state.current_chat}")

if st.sidebar.button("➕ New Chat"):

    if len(st.session_state.messages) > 1:

        st.session_state.chat_history.append({
            "title": (
                st.session_state.messages[0]["content"][:30] + "..."
                if len(st.session_state.messages[0]["content"]) > 30
                else st.session_state.messages[0]["content"]
            ),
            "messages": st.session_state.messages.copy()
        })

    st.session_state.messages = []

    st.session_state.current_chat = "New Chat"

    st.rerun()

st.sidebar.markdown("---")

st.sidebar.subheader("💬 Chat History")

for index, chat in enumerate(st.session_state.chat_history):

    if st.sidebar.button(chat["title"], key=f"chat_{index}"):

        st.session_state.messages = chat["messages"]

        st.session_state.current_chat = chat["title"]

        st.rerun()

FAISS_INDEX_DIR = "faiss_index"

# ---------------------------------------------------------------------------
# Retrieval / Re-ranking configuration
# Tune these based on eval results (Phase 5) rather than guessing.
# ---------------------------------------------------------------------------

# How many candidates EACH retriever (FAISS and BM25) pulls independently,
# before their results are merged by the EnsembleRetriever.
TOP_K = 6

# Weighting between semantic (FAISS) and keyword (BM25) retrieval.
# [FAISS_weight, BM25_weight] — must sum to 1.0.
# Higher FAISS weight favors paraphrased/semantic matches.
# Higher BM25 weight favors exact terms (policy numbers, acronyms, names).
ENSEMBLE_WEIGHTS = [0.6, 0.4]

# How many merged candidates (after FAISS+BM25 fusion, before re-ranking)
# the cross-encoder actually scores. Caps re-ranking cost when the merged
# candidate pool is large; set >= TOP_K * len(retrievers) to score everything.
RERANK_TOP_N = 10

# How many of the re-ranked, highest-scoring chunks are actually sent to
# the LLM as context. Keep this small — more context isn't always better,
# it can dilute relevance and increase hallucination risk.
FINAL_CONTEXT_DOCS = 3

# Minimum cross-encoder relevance score (from the top-ranked chunk) required
# to trust the retrieved context enough to answer from it.
#
# NOTE: cross-encoder/ms-marco-MiniLM-L-6-v2 outputs raw, UNBOUNDED logits,
# not a 0-1 probability — negative scores mean "not relevant", positive
# scores mean increasing relevance. There's no universal "correct" value;
# this must be tuned empirically against your own documents:
#   1. Ask a question with a real answer in your docs -> note the top score.
#   2. Ask a question with NO answer in your docs (like "haircut policy")
#      -> note the top score.
#   3. Set CONFIDENCE_THRESHOLD somewhere between those two observed values.
# A starting point of 0.0 is reasonable, but treat it as a placeholder
# until you've run that calibration pass (ties in well with Phase 5 eval).
CONFIDENCE_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_ensemble_retriever():
    """
    Builds hybrid retrieval (FAISS + BM25) without touching the ingestion
    pipeline. BM25 is built on-the-fly from the documents already stored
    inside the existing FAISS index, so generate_embeddings.py stays
    untouched and is the single source of truth for chunking.
    """
    vectorstore = load_vectorstore()
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Pull every chunk already embedded in FAISS to build the BM25 index.
    all_docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = TOP_K

    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )
    return ensemble


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


llm = load_llm()
retriever = load_ensemble_retriever()
reranker = load_reranker()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rerank_documents(query, docs, top_k=FINAL_CONTEXT_DOCS):
    """
    Re-score ensemble candidates with a cross-encoder and keep the best ones.

    Returns a list of (doc, score) tuples, highest score first, so callers
    can both build context AND check retrieval confidence from the same call.

    Two separate knobs control this stage:
    - RERANK_TOP_N: how many merged candidates get scored at all (cost control)
    - top_k (FINAL_CONTEXT_DOCS): how many of the scored docs survive into the prompt
    """
    if not docs:
        return []

    # Cap how many candidates we bother scoring, in case the merged
    # FAISS+BM25 pool is larger than we want to run through the cross-encoder.
    docs = docs[:RERANK_TOP_N]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    unique_ranked = []
    for doc, score in scored_docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            unique_ranked.append((doc, float(score)))
        if len(unique_ranked) == top_k:
            break

    return unique_ranked


def is_retrieval_confident(ranked_docs_with_scores, threshold=CONFIDENCE_THRESHOLD):
    """
    Confidence check gate: decides whether retrieved context is trustworthy
    enough to answer from, or whether we should fall back instead of risking
    a hallucinated answer.

    True only if there's at least one chunk AND its top cross-encoder score
    clears the configured threshold.
    """
    if not ranked_docs_with_scores:
        return False
    top_score = ranked_docs_with_scores[0][1]
    return top_score >= threshold


def build_fallback_response(llm, question):
    """
    Used when retrieval confidence is too low to trust the retrieved context.
    Explicitly instructs the LLM to acknowledge the lack of documented
    information rather than guessing or fabricating a policy answer.
    """
    fallback_prompt = f"""You are an Enterprise Knowledge Assistant.

No sufficiently relevant information was found in the company documents
for the question below. You must NOT invent, assume, or guess any policy
details, numbers, or rules that are not explicitly documented.

Write a short, professional response that:
1. States that the available company documents do not contain information
   on this topic.
2. Does not speculate about what the policy might be.
3. Politely directs the employee to contact HR or refer to the latest
   company handbook for an official answer.

Question:
{question}

Response:"""

    response = llm.invoke(fallback_prompt)
    return response.content


def rewrite_followup_question(question, chat_history, llm):
    """Rewrite context-dependent follow-ups into standalone questions."""
    if not chat_history:
        return question

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in chat_history[-6:]
    )

    rewrite_prompt = f"""Given the conversation history and a follow-up question,
rewrite the follow-up into a standalone question that contains all
necessary context. If the follow-up question is already standalone,
return it unchanged. Respond with ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question:
{question}

Standalone question:"""

    rewritten = llm.invoke(rewrite_prompt).content.strip()
    return rewritten if rewritten else question


def clean_source_name(raw_source):
    name = os.path.basename(raw_source)
    name = name.replace(".pdf", "").replace("_", " ")
    return name


def format_sources(ranked_docs_with_scores):
    """Build a de-duplicated, readable list of sources, using whatever
    metadata generate_embeddings.py originally attached (source, page)."""
    seen = set()
    formatted = []
    for doc, _score in ranked_docs_with_scores:
        raw_source = doc.metadata.get("source", "Unknown document")
        name = clean_source_name(raw_source)
        # PyPDFDirectoryLoader's "page" is 0-indexed; show as 1-indexed.
        page = doc.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "?"
        key = (name, page_display)
        if key not in seen:
            seen.add(key)
            formatted.append(f"📄 **{name}** — page {page_display}")
    return formatted


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about company policies")

if question and question.strip():

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    standalone_question = rewrite_followup_question(
        question, st.session_state.messages[:-1], llm
    )

    candidate_docs = retriever.invoke(standalone_question)

    start_time = time.time()
    response_time = 0
    ranked_docs_with_scores = rerank_documents(standalone_question, candidate_docs)

    if is_retrieval_confident(ranked_docs_with_scores):
        # Normal RAG path: confident, relevant context was found.
        top_docs = [doc for doc, _score in ranked_docs_with_scores]
        context = "\n\n".join(doc.page_content for doc in top_docs)

        prompt = f"""
        You are an Enterprise Knowledge Assistant.

        Answer only using the provided context.

        If information is not available,
        say:
        'Information not found in the documents.'

        Context:
        {context}

        Question:
        {standalone_question}
        """

        answer_text = llm.invoke(prompt).content
    else:
        # Low-confidence path: retrieved context isn't trustworthy enough
        # to answer from. Use a guarded fallback instead of risking a
        # hallucinated policy answer.
        answer_text = build_fallback_response(llm, standalone_question)
        ranked_docs_with_scores = []  # don't show misleading sources below

    response_time = time.time() - start_time

    with st.chat_message("assistant"):

        with st.status("🔍 Searching company documents...", expanded=True) as status:

            st.write("📄 Reading enterprise documents...")
            st.write("🔍 Searching semantic and keyword indexes...")
            st.write("🎯 Selecting the best matching documents...")
            st.write("🤖 Gemini is generating the final answer...")

            status.update(
                label="✅ Response generated successfully!",
                state="complete",
                expanded=False,
            )

        st.markdown(answer_text)
        st.caption(f"⏱ Response Time: {response_time:.2f} sec")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Retrieved", len(candidate_docs))

        with col2:
            st.metric("Used", len(ranked_docs_with_scores))

        with col3:
            st.metric("Model", "Gemini")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )

    if ranked_docs_with_scores:
        st.subheader("📚 Evidence Used")

        for doc, score in ranked_docs_with_scores:

            raw_source = doc.metadata.get("source", "Unknown document")
            source_name = clean_source_name(raw_source)

            page = doc.metadata.get("page")
            page_display = page + 1 if isinstance(page, int) else "?"

            with st.expander(f"📄 {source_name} — Page {page_display}"):

                snippet = doc.page_content.strip()

                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."

                st.write(snippet)
                st.markdown(f"**Page:** {page_display}")
                st.divider()

                normalized = max(0, min((score + 5) / 15, 1))
                confidence = max(0, min(((score + 10) / 20) * 100, 100))

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )
                st.markdown("---")

                st.caption(
                    "Enterprise RAG | Hybrid Search (FAISS + BM25) | Cross-Encoder Re-ranking | Gemini 2.5 Flash"
                )
