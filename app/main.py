import sys
from pathlib import Path
import streamlit as st
import tempfile
import os

# ---------------------------------------------------------
# Fix project root imports
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------
# Ingestion
# ---------------------------------------------------------
from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text

# ---------------------------------------------------------
# Processing (BATCHED)
# ---------------------------------------------------------
from processing.chunker import chunk_document
from processing.batch_classifier import classify_clauses_batch
from processing.batch_simplifier import simplify_clauses_batch
from processing.batch_risk_analyzer import analyze_risks_batch
from processing.aggregator import aggregate_document
from processing.answer_synthesizer import synthesize_answer

# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------
from retrieval.vectorstore import create_vectorstore
from retrieval.multi_query import multi_query_retrieve


# =========================================================
# Streamlit Page Config
# =========================================================
st.set_page_config(
    page_title="GenAI Legal Document Explainer",
    layout="wide"
)

st.title("📄 GenAI Legal Document Explainer")
st.caption("⚠️ This tool provides general information and is not legal advice.")

# =========================================================
# Session State Initialization
# =========================================================
if "processed" not in st.session_state:
    st.session_state.processed = False

if "clauses" not in st.session_state:
    st.session_state.clauses = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_insights" not in st.session_state:
    st.session_state.document_insights = None


# =========================================================
# File Upload
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a legal document (PDF)",
    type=["pdf"]
)

# =========================================================
# Analyze Button (Guarded)
# =========================================================
if uploaded_file and st.button("Analyze Document") and not st.session_state.processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Processing document..."):
        # ---- Ingest ----
        raw_result = load_pdf(pdf_path, source="user")
        cleaned_text = clean_text(raw_result["text"])

        # ---- Chunk ----
        clauses = chunk_document(cleaned_text)

        # ---- Batched LLM processing ----
        classification_map = classify_clauses_batch(clauses)
        simplification_map = simplify_clauses_batch(clauses)
        risk_map = analyze_risks_batch(clauses)

        # ---- Attach results safely ----
        for clause in clauses:
            cid = clause["chunk_id"]

            clause["categories"] = classification_map.get(cid, {}).get(
                "categories", ["general"]
            )

            clause["simplified"] = simplification_map.get(cid, {
                "simple_explanation": "",
                "user_impact": "",
                "key_points": []
            })

            clause["risk"] = risk_map.get(cid, {
                "risk_level": "low",
                "risk_types": [],
                "risk_summary": "",
                "red_flags": []
            })

        # ---- Aggregate ----
        document_insights = aggregate_document(clauses)

        # ---- Vector Store ----
        vectorstore = create_vectorstore(clauses)

        # ---- Save to session ----
        st.session_state.clauses = clauses
        st.session_state.vectorstore = vectorstore
        st.session_state.document_insights = document_insights
        st.session_state.processed = True

    os.remove(pdf_path)
    st.success("Document analysis completed successfully")

# =========================================================
# Stop if not processed
# =========================================================
if not st.session_state.processed:
    st.info("Upload a document and click **Analyze Document** to begin.")
    st.stop()

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["📊 Document Overview", "📑 Clause Breakdown", "❓ Ask a Question"]
)

# =========================================================
# TAB 1: Document Overview
# =========================================================
with tab1:
    insights = st.session_state.document_insights

    st.subheader("Overall Risk Assessment")

    st.metric(
        label="Overall Risk Level",
        value=insights["overall_risk_level"].upper()
    )

    st.progress(insights["risk_score"] / 100)
    st.caption(f"Risk Score: {insights['risk_score']} / 100")

    st.subheader("Clause Category Distribution")
    st.json(insights["category_distribution"])

    if insights["top_risky_clauses"]:
        st.subheader("⚠️ High Risk Clauses")
        for rc in insights["top_risky_clauses"]:
            st.markdown(
                f"**Clause {rc['chunk_id']} – {rc['heading']}**  \n"
                f"{rc['risk_summary']}"
            )

# =========================================================
# TAB 2: Clause Breakdown
# =========================================================
with tab2:
    for clause in st.session_state.clauses:
        with st.expander(
            f"Clause {clause['chunk_id']} – {clause.get('heading', 'General')}"
        ):
            st.markdown("**Original Clause**")
            st.write(clause["text"])

            st.markdown("**Categories**")
            st.write(", ".join(clause["categories"]))

            st.markdown("**Simplified Explanation**")
            st.write(clause["simplified"]["simple_explanation"])

            st.markdown("**User Impact**")
            st.write(clause["simplified"]["user_impact"])

            st.markdown("**Risk Analysis**")
            st.write(f"Risk Level: {clause['risk']['risk_level'].upper()}")
            st.write(clause["risk"]["risk_summary"])

# =========================================================
# TAB 3: Ask a Question
# =========================================================
with tab3:
    user_question = st.text_input(
        "Ask a question about this document"
    )

    if user_question:
        with st.spinner("Retrieving relevant clauses..."):
            retrieved = multi_query_retrieve(
                st.session_state.vectorstore,
                user_query=user_question
            )

        with st.spinner("Generating grounded answer..."):
            answer = synthesize_answer(
                question=user_question,
                retrieved_clauses=retrieved
            )

        st.subheader("Answer")
        st.write(answer["answer"])

        with st.expander("🔍 Supporting Clauses"):
            for r in retrieved:
                meta = r["metadata"]
                st.markdown(
                    f"**Clause {meta.get('chunk_id')} – {meta.get('heading')}**"
                )
                st.write(r["text"])
