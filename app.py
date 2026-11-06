"""
app.py - Streamlit Web Application (RAG-15, RAG-16)
Full-Featured Interactive Web UI for the Local LLM College Document Q&A Assistant
"""

import os
import time
import json
import streamlit as st

from rag_service import RAGService
from llm_service import is_ollama_available
from vector_store import VectorStore
from generate_sample_pdfs import create_all_college_pdfs

# Page configuration
st.set_page_config(
    page_title="College Document Q&A Assistant (RAG)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-online {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: #FEECDC;
        color: #9C4221;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-pill {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #C7D2FE;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 16px 20px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .refusal-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag_service: RAGService = st.session_state.rag_service
v_store: VectorStore = rag_service.vector_store
stats = v_store.get_stats()
ollama_online = is_ollama_available()

# Check LLM Engine status
from llm_service import get_active_api_key
cloud_provider, cloud_key = get_active_api_key()

# Sidebar: System Status & Quick Actions
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("System Status")

    # Engine Status
    if cloud_provider:
        st.markdown(f'<span class="badge-online">● Cloud API: {cloud_provider.upper()}</span>', unsafe_allow_html=True)
        st.caption("Connected via API Key")
    elif ollama_online:
        st.markdown('<span class="badge-online">● Ollama LLM: Online</span>', unsafe_allow_html=True)
        st.caption("Connected to localhost:11434 (`llama3.2:3b`)")
    else:
        st.markdown('<span class="badge-offline">○ Local Fallback: Active</span>', unsafe_allow_html=True)
        st.caption("Running with Local Extractive Grounding.")

    # Optional API Key Config
    with st.expander("🔑 Cloud API Settings (Optional)"):
        st.caption("No API key is required (₹0 stack). If you have a free Google Gemini API key or Groq key, enter it below:")
        api_choice = st.selectbox("API Provider", ["Google Gemini", "Groq", "OpenAI"])
        api_val = st.text_input("Enter API Key:", type="password")
        if st.button("Save API Key"):
            if api_val.strip():
                env_var = "GEMINI_API_KEY" if api_choice == "Google Gemini" else ("GROQ_API_KEY" if api_choice == "Groq" else "OPENAI_API_KEY")
                os.environ[env_var] = api_val.strip()
                with open(".env", "a", encoding="utf-8") as f:
                    f.write(f"\n{env_var}={api_val.strip()}\n")
                st.success(f"{api_choice} Key Saved!")
                st.rerun()

    st.divider()

    st.subheader("📚 Knowledge Base")
    st.metric("Total Indexed Chunks", stats["total_chunks"])
    st.metric("Indexed Documents", len(stats["unique_documents"]))

    if stats["unique_documents"]:
        with st.expander("Active Documents"):
            for doc in stats["unique_documents"]:
                st.write(f"- 📄 `{doc}`")

    st.divider()

    if st.button("🔄 Auto-Generate Sample PDFs", use_container_width=True):
        with st.spinner("Generating official college PDFs..."):
            create_all_college_pdfs("./data")
            ingest_res = rag_service.ingest_documents_from_directory("./data")
            st.success(f"Indexed {ingest_res.get('chunks_added', 0)} chunks from college documents!")
            st.rerun()

    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        v_store.clear()
        st.warning("Vector database cleared.")
        st.rerun()

    st.divider()
    st.caption("Final-Year Mini Project | ₹0 Zero-Cost Local RAG Architecture")


# Main Interface Tabs
tab_qa, tab_docs, tab_eval, tab_jira, tab_guide = st.tabs([
    "💬 Student Q&A Assistant",
    "📂 Document Repository",
    "📊 Evaluation & Benchmarks",
    "📋 Jira Scrum Board",
    "📖 Architecture & Setup Guide"
])


# ==========================================
# TAB 1: Student Q&A Assistant
# ==========================================
with tab_qa:
    st.markdown('<div class="main-title">College Document Question Answering Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Retrieval-Augmented Generation (RAG) over official college regulations, handbooks, and policies.</div>', unsafe_allow_html=True)

    if stats["total_chunks"] == 0:
        st.info("💡 The vector database is currently empty. Click the button below to generate and index the sample college PDF documents.", icon="ℹ️")
        if st.button("📥 Index Official College Documents Now", type="primary"):
            create_all_college_pdfs("./data")
            res = rag_service.ingest_documents_from_directory("./data")
            st.success(f"Successfully processed and indexed {res.get('chunks_added', 0)} chunks!")
            st.rerun()

    # Sample Quick Questions
    st.write("**Quick Questions:**")
    quick_cols = st.columns(4)
    sample_queries = [
        "What is the minimum attendance required?",
        "How much is the fee for examination revaluation?",
        "What is the CGPA eligibility for campus placements?",
        "What is the annual hostel fee?"  # Unanswerable test
    ]

    selected_query = None
    for i, col in enumerate(quick_cols):
        if col.button(sample_queries[i], key=f"quick_{i}"):
            selected_query = sample_queries[i]

    # Input Box
    query_text = st.text_input(
        "Enter your question regarding college regulations:",
        value=selected_query if selected_query else "",
        placeholder="e.g., How many credits are required for graduation?"
    )

    col_btn, col_k = st.columns([1, 4])
    ask_clicked = col_btn.button("🔍 Ask Question", type="primary", use_container_width=True)
    top_k = col_k.slider("Top Chunks to Retrieve (k)", min_value=1, max_value=6, value=3)

    if ask_clicked and query_text:
        with st.spinner("Retrieving relevant passages and formulating answer..."):
            start_time = time.time()
            response = rag_service.answer_question(query_text, top_k=top_k)
            latency = round(time.time() - start_time, 2)

            answer = response["answer"]
            sources = response["sources"]
            retrieved_chunks = response["retrieved_chunks"]
            is_refusal = response["is_safe_refusal"]

            # Save in history
            st.session_state.chat_history.insert(0, {
                "question": query_text,
                "answer": answer,
                "sources": sources,
                "chunks": retrieved_chunks,
                "is_refusal": is_refusal,
                "engine": response.get("engine", "Local"),
                "latency": latency
            })

    # Display Response
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[0]
        st.divider()
        st.subheader("💡 Answer")

        if latest["is_refusal"] or "information not found" in latest["answer"].lower():
            st.markdown(f'<div class="refusal-box">⚠️ <strong>{latest["answer"]}</strong></div>', unsafe_allow_html=True)
            st.caption("🛡️ **Hallucination Prevention Active**: The query requested details not present in the ingested college documents. The system safely declined to invent facts.")
        else:
            st.markdown(f'<div class="answer-box">{latest["answer"]}</div>', unsafe_allow_html=True)

            if latest["sources"]:
                st.write("**Verified Sources:**")
                for s in latest["sources"]:
                    st.markdown(f'<span class="source-pill">📄 {s}</span>', unsafe_allow_html=True)

        st.caption(f"⚡ Generated by **{latest['engine']}** in `{latest['latency']}s`")

        # Inspect retrieved chunks
        with st.expander("🔍 Inspect Retrieved Context Chunks (ChromaDB)", expanded=False):
            if not latest["chunks"]:
                st.write("No relevant chunks retrieved.")
            for i, chunk in enumerate(latest["chunks"], start=1):
                st.markdown(f"**Chunk {i}** | Source: `{chunk['source']}` (Page {chunk['page']}) | Similarity Score: `{chunk['similarity']}`")
                st.info(chunk["text"])


# ==========================================
# TAB 2: Document Repository
# ==========================================
with tab_docs:
    st.subheader("📁 Upload and Manage College Documents")
    st.write("Upload official PDF documents (Syllabus, Regulations, Handbooks, Notices) to build or expand the local RAG knowledge base.")

    uploaded_files = st.file_uploader("Choose PDF files to ingest", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("🚀 Ingest & Index Uploaded PDFs", type="primary"):
        with st.spinner("Processing documents..."):
            os.makedirs("./data", exist_ok=True)
            total_added = 0
            for uploaded_file in uploaded_files:
                save_path = os.path.join("./data", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                res = rag_service.ingest_single_file(save_path)
                total_added += res.get("chunks_added", 0)

            st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into {total_added} vector chunks!")
            st.rerun()

    st.divider()
    st.subheader("📑 Documents in Local Storage (`./data`)")
    if os.path.exists("./data"):
        data_files = [f for f in os.listdir("./data") if f.lower().endswith(".pdf")]
        if data_files:
            for f in data_files:
                f_path = os.path.join("./data", f)
                f_size = round(os.path.getsize(f_path) / 1024, 1)
                c1, c2 = st.columns([4, 1])
                c1.write(f"📄 **{f}** ({f_size} KB)")
                if c2.button("Re-index", key=f"reindex_{f}"):
                    rag_service.ingest_single_file(f_path)
                    st.success(f"Re-indexed `{f}`")
                    st.rerun()

                # Document text preview expander
                from document_loader import read_pdf
                with st.expander(f"👁️ Preview Text: {f}"):
                    pages = read_pdf(f_path)
                    for p in pages:
                        st.markdown(f"**Page {p['page']} of {p['total_pages']}**")
                        st.text(p["text"])
        else:
            st.write("No PDF documents found in `./data`.")


# ==========================================
# TAB 3: Evaluation & Benchmarks
# ==========================================
with tab_eval:
    st.subheader("📊 RAG Evaluation & Testing Framework (STEP 21, 22, 23)")
    st.write("Run automated evaluation across **50 benchmark questions** partitioned into Direct Questions (Category A), Paraphrased Questions (Category B), and Unanswerable Probes (Category C).")

    if st.button("▶️ Run 50-Question Benchmark", type="primary"):
        from tests.evaluate_rag import run_evaluation
        with st.spinner("Evaluating 50 questions against local RAG pipeline..."):
            report = run_evaluation()
            st.session_state.eval_report = report
            st.success("Benchmark completed successfully!")

    if "eval_report" in st.session_state:
        rep = st.session_state.eval_report

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval Accuracy", f"{rep['retrieval_accuracy_percent']}%", f"{rep['correct_retrievals']}/{rep['answerable_questions']} correct")
        col2.metric("Safe Refusal Rate", f"{rep['safe_refusal_rate_percent']}%", f"{rep['correct_safe_refusals']}/{rep['unanswerable_questions']} unanswerable")
        col3.metric("Total Questions", rep["total_questions"])
        col4.metric("Avg Latency", f"{rep['average_latency_seconds']}s")

        st.divider()
        c_head, c_dl = st.columns([3, 1])
        c_head.subheader("Detailed Evaluation Results")
        
        # Download button for evaluation report
        eval_csv_lines = ["ID,Category,Question,Expected Document,Retrieved Documents,Retrieval Success,Safe Refusal,Latency (s)"]
        for it in rep["results"]:
            q_clean = it["question"].replace('"', '""')
            ret_docs = " | ".join(it["retrieved_documents"]) if it["retrieved_documents"] else "None"
            succ = "Yes" if it["retrieval_success"] else ("N/A" if not it["is_answerable"] else "No")
            ref = "Yes" if it["is_safe_refusal"] else ("No" if not it["is_answerable"] else "-")
            eval_csv_lines.append(f'{it["id"]},"{it["category"]}","{q_clean}","{it["expected_document"]}","{ret_docs}",{succ},{ref},{it["latency_sec"]}')
        
        c_dl.download_button(
            label="📥 Download Evaluation CSV",
            data="\n".join(eval_csv_lines),
            file_name="rag_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        filter_cat = st.selectbox("Filter by Category", ["All", "A - Direct", "B - Paraphrased", "C - Unanswerable"])
        items = rep["results"]
        if filter_cat != "All":
            items = [i for i in items if filter_cat in i["category"]]

        table_data = []
        for it in items:
            table_data.append({
                "ID": it["id"],
                "Category": it["category"],
                "Question": it["question"],
                "Expected Document": it["expected_document"],
                "Retrieved Document(s)": ", ".join(it["retrieved_documents"]) if it["retrieved_documents"] else "None",
                "Retrieval Success": "✅ Yes" if it["retrieval_success"] else ("N/A" if not it["is_answerable"] else "❌ No"),
                "Safe Refusal": "🛡️ Yes" if it["is_safe_refusal"] else ("No" if not it["is_answerable"] else "-"),
                "Latency (s)": it["latency_sec"]
            })
        st.dataframe(table_data, use_container_width=True)


# ==========================================
# TAB 4: Jira Project Management Hub
# ==========================================
with tab_jira:
    col_jhead, col_jdl = st.columns([3, 1])
    col_jhead.subheader("📋 Jira Project Management & Tracking (STEP 1 - STEP 6)")
    col_jhead.write("Project Name: **College RAG Assistant** | Project Key: **RAG** | Methodology: **Scrum**")

    # Read Jira CSV for export
    jira_csv_path = "./jira_export.csv"
    if os.path.exists(jira_csv_path):
        with open(jira_csv_path, "r", encoding="utf-8") as jf:
            jira_csv_data = jf.read()
        col_jdl.download_button(
            label="📥 Export Jira Backlog CSV",
            data=jira_csv_data,
            file_name="jira_college_rag_backlog.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Sprint filter
    sprint_choice = st.radio(
        "Select Sprint View:",
        ["All Sprints", "Sprint 1 — Foundation", "Sprint 2 — RAG Pipeline", "Sprint 3 — LLM Application", "Sprint 4 — Testing & Submission"],
        horizontal=True
    )

    # 20 Jira Tasks Specification
    tasks_data = [
        {"key": "RAG-1", "title": "Define functional requirements", "epic": "EPIC-1", "sp": 3, "sprint": "Sprint 1", "status": "DONE", "assignee": "Student 1"},
        {"key": "RAG-2", "title": "Collect college documents (5 PDFs)", "epic": "EPIC-1", "sp": 3, "sprint": "Sprint 1", "status": "DONE", "assignee": "Student 1"},
        {"key": "RAG-3", "title": "Prepare 50 sample questions (A, B, C)", "epic": "EPIC-1", "sp": 5, "sprint": "Sprint 1", "status": "DONE", "assignee": "Student 4"},
        {"key": "RAG-4", "title": "Install Ollama runtime", "epic": "EPIC-2", "sp": 2, "sprint": "Sprint 1", "status": "DONE", "assignee": "Student 3"},
        {"key": "RAG-5", "title": "Run and test local LLM (llama3.2:3b)", "epic": "EPIC-2", "sp": 3, "sprint": "Sprint 1", "status": "DONE", "assignee": "Student 3"},

        {"key": "RAG-6", "title": "Develop PDF text extractor (PyPDF)", "epic": "EPIC-3", "sp": 3, "sprint": "Sprint 2", "status": "DONE", "assignee": "Student 1"},
        {"key": "RAG-7", "title": "Develop text chunking module", "epic": "EPIC-3", "sp": 3, "sprint": "Sprint 2", "status": "DONE", "assignee": "Student 1"},
        {"key": "RAG-8", "title": "Generate embeddings (nomic-embed-text)", "epic": "EPIC-3", "sp": 5, "sprint": "Sprint 2", "status": "DONE", "assignee": "Student 2"},
        {"key": "RAG-9", "title": "Create vector database (ChromaDB)", "epic": "EPIC-3", "sp": 5, "sprint": "Sprint 2", "status": "DONE", "assignee": "Student 2"},
        {"key": "RAG-10", "title": "Develop semantic retrieval module", "epic": "EPIC-3", "sp": 5, "sprint": "Sprint 2", "status": "DONE", "assignee": "Student 2"},

        {"key": "RAG-11", "title": "Integrate retrieved context with LLM", "epic": "EPIC-3", "sp": 5, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 3"},
        {"key": "RAG-12", "title": "Develop RAG prompt template", "epic": "EPIC-3", "sp": 3, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 3"},
        {"key": "RAG-13", "title": "Implement source citation", "epic": "EPIC-3", "sp": 3, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 2"},
        {"key": "RAG-14", "title": "Implement unknown-answer handling", "epic": "EPIC-3", "sp": 3, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 3"},
        {"key": "RAG-15", "title": "Create Streamlit UI", "epic": "EPIC-4", "sp": 5, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 4"},
        {"key": "RAG-16", "title": "Display retrieved sources & metadata", "epic": "EPIC-4", "sp": 3, "sprint": "Sprint 3", "status": "DONE", "assignee": "Student 4"},

        {"key": "RAG-17", "title": "Perform functional testing", "epic": "EPIC-5", "sp": 5, "sprint": "Sprint 4", "status": "DONE", "assignee": "Student 4"},
        {"key": "RAG-18", "title": "Evaluate LLM responses & accuracy", "epic": "EPIC-5", "sp": 5, "sprint": "Sprint 4", "status": "DONE", "assignee": "Student 2"},
        {"key": "RAG-19", "title": "Fix identified defects & edge cases", "epic": "EPIC-5", "sp": 5, "sprint": "Sprint 4", "status": "DONE", "assignee": "Student 3"},
        {"key": "RAG-20", "title": "Prepare documentation and demo", "epic": "EPIC-6", "sp": 5, "sprint": "Sprint 4", "status": "DONE", "assignee": "All Students"}
    ]

    filtered_tasks = tasks_data
    if sprint_choice != "All Sprints":
        sprint_name = sprint_choice.split(" — ")[0]
        filtered_tasks = [t for t in tasks_data if t["sprint"] == sprint_name]

    # Kanban / Scrum Board Columns
    col_todo, col_inprog, col_review, col_test, col_done = st.columns(5)
    cols = {
        "TO DO": col_todo,
        "IN PROGRESS": col_inprog,
        "CODE REVIEW": col_review,
        "TESTING": col_test,
        "DONE": col_done
    }

    col_todo.markdown("##### 📌 TO DO")
    col_inprog.markdown("##### ⚙️ IN PROGRESS")
    col_review.markdown("##### 🔍 CODE REVIEW")
    col_test.markdown("##### 🧪 TESTING")
    col_done.markdown("##### ✅ DONE")

    for task in filtered_tasks:
        target_col = cols.get(task["status"], col_done)
        with target_col:
            st.markdown(f"""
            <div style="background-color: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 6px; padding: 10px; margin-bottom: 10px;">
                <strong>{task['key']}</strong> <span style="font-size:0.75rem; background:#E5E7EB; padding:2px 6px; border-radius:4px;">{task['sp']} SP</span><br/>
                <span style="font-size:0.9rem;">{task['title']}</span><br/>
                <span style="font-size:0.75rem; color:#6B7280;">👤 {task['assignee']} | 🏷️ {task['epic']}</span>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# TAB 5: Architecture & Setup Guide
# ==========================================
with tab_guide:
    st.subheader("📖 Zero-Cost Local RAG Architecture & Student Setup Guide")
    st.markdown("""
### 1. Zero-Cost Technology Stack Overview
| Component | Technology | Cost | Description |
|---|---|---|---|
| **Project Tracking** | Atlassian Jira Free | ₹0 | Scrum workflow, sprints, and bug tracking |
| **Language & Framework** | Python 3.10+ | ₹0 | Core backend development |
| **Local LLM Runtime** | Ollama | ₹0 | Runs models locally on CPU or GPU |
| **Chat LLM Model** | `llama3.2:3b` | ₹0 | Generates answers strictly from context |
| **Embeddings Model** | `nomic-embed-text` | ₹0 | 768-dim high quality embeddings |
| **Vector Store** | ChromaDB | ₹0 | Embedded vector database with persistence |
| **PDF Extraction** | PyPDF | ₹0 | Extracts clean text and metadata from PDFs |
| **User Interface** | Streamlit | ₹0 | Responsive web application |
| **Hosting & API** | Local Computer | ₹0 | No cloud API fees (OpenAI / Claude / Gemini ₹0) |

---

### 2. How to Run Locally (Step-by-Step)

#### Step A: Install and Start Ollama
1. Download Ollama from: [https://ollama.com](https://ollama.com)
2. In your terminal, pull the zero-cost models:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```
3. Test your model:
```bash
ollama run llama3.2:3b "Explain RAG in simple language"
```

#### Step B: Launch the Web Assistant
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
""")
