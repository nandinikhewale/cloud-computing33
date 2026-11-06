# College Document Question Answering Assistant Using Local RAG & Jira

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Vector Store](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Local LLM](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com/)
[![Cost](https://img.shields.io/badge/API_Cost-₹0_Free-brightgreen.svg)]()

> **Final-Year Mini Project**: Design and Development of a Local LLM-Based College Document Question Answering Assistant Using Retrieval-Augmented Generation (RAG) and Project Management with Jira.

---

## 📌 Project Overview
Colleges maintain large amounts of critical information in varied PDF documents (academic regulations, syllabus books, examination guidelines, attendance rules, placement policies, etc.). Students frequently have questions such as:
- *What is the minimum attendance required to appear for examinations?*
- *How much is the fee for examination revaluation?*
- *What is the CGPA eligibility criterion for campus placement drives?*

Instead of manually leafing through 100+ pages of PDFs, this application provides an **intelligent local Question Answering Assistant powered by Retrieval-Augmented Generation (RAG)**:
1. **Reads & Chunks** college PDFs while retaining document names and page numbers.
2. **Generates Vector Embeddings** locally (`nomic-embed-text`).
3. **Stores & Indexes Chunks** locally in ChromaDB.
4. **Semantically Retrieves** the top matching sections upon user query.
5. **Generates Grounded Answers** using a local LLM (`llama3.2:3b`).
6. **Cites Sources** (document name & page number).
7. **Prevents Hallucinations** by safely declining when information is not in the documents (*"Information not found in the available documents."*).

---

## 💰 Zero-Cost Technology Stack & API Requirements

### ❓ What APIs Are Required?
**None! Zero external paid APIs are required.**
- ❌ OpenAI API: Not required (₹0)
- ❌ Gemini API: Not required (₹0)
- ❌ Claude API: Not required (₹0)
- ❌ Cloud Database Hosting: Not required (₹0)

The entire software pipeline runs **100% locally** on student laptops using:
- **Programming**: Python 3.10+
- **LLM Runtime**: Ollama (Free & Open Source)
- **Chat LLM**: `llama3.2:3b`
- **Embedding Model**: `nomic-embed-text`
- **Vector Database**: ChromaDB (Embedded local persistent vector DB)
- **User Interface**: Streamlit
- **PDF Extraction**: PyPDF
- **Project Tracking**: Atlassian Jira Free

---

## 🚀 Quick Start Guide

### 1. Install Python Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Setup Ollama (Local LLM & Embeddings)
1. Download & install Ollama from [https://ollama.com](https://ollama.com).
2. Pull the required models in your terminal:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```
3. Test your LLM:
```bash
ollama run llama3.2:3b "Explain RAG in simple language"
```

### 3. Generate College PDF Documents
Run the sample PDF generator to generate 5 official college documents in `./data/`:
```bash
python generate_sample_pdfs.py
```
Generated documents:
- `data/academic_regulations.pdf`
- `data/examination_guidelines.pdf`
- `data/student_handbook.pdf`
- `data/placement_guidelines.pdf`
- `data/department_rules.pdf`

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Evaluation & Testing Framework (50-Question Benchmark)
To test retrieval quality and hallucination prevention, run the automated benchmark:
```bash
python tests/evaluate_rag.py
```
Or view the interactive evaluation dashboard in the **Evaluation & Benchmarks** tab of the web app.

### Evaluation Metrics:
$$\text{Retrieval Accuracy} = \frac{\text{Correct Retrievals}}{\text{Total Answerable Questions}} \times 100$$
$$\text{Safe Refusal Rate} = \frac{\text{Correct Refusals for Unanswerable Questions}}{\text{Total Unanswerable Questions}} \times 100$$

---

## 📋 Jira Project Management & Tracking

### Jira Setup
- **Project Name**: College RAG Assistant
- **Project Key**: `RAG`
- **Project Type**: Scrum
- **Workflow**: `TO DO` → `IN PROGRESS` → `CODE REVIEW` → `TESTING` → `DONE`

### Epics
- `EPIC-1`: Requirements and Dataset
- `EPIC-2`: Local LLM Setup
- `EPIC-3`: RAG Pipeline
- `EPIC-4`: User Interface
- `EPIC-5`: Testing and Evaluation
- `EPIC-6`: Documentation and Demonstration

### 4-Sprint Plan
| Sprint | Backlog Items | Goal |
|---|---|---|
| **Sprint 1 (Foundation)** | RAG-1 to RAG-5 | Environment setup, 5 PDFs, 50 questions, Ollama installation |
| **Sprint 2 (RAG Pipeline)** | RAG-6 to RAG-10 | PDF extraction, chunking, embeddings, ChromaDB, semantic retrieval |
| **Sprint 3 (LLM Application)** | RAG-11 to RAG-16 | Context prompt integration, citation, refusal, Streamlit UI |
| **Sprint 4 (Testing & Demo)** | RAG-17 to RAG-20 | Evaluation benchmarks, bug fixes, final project report & presentation |

### Complete 20-Item Backlog
| ID | Title | Epic | Story Points | Assignee |
|---|---|---|---|---|
| **RAG-1** | Define functional requirements | EPIC-1 | 3 | Student 1 |
| **RAG-2** | Collect college documents | EPIC-1 | 3 | Student 1 |
| **RAG-3** | Prepare 50 sample questions | EPIC-1 | 5 | Student 4 |
| **RAG-4** | Install Ollama runtime | EPIC-2 | 2 | Student 3 |
| **RAG-5** | Run and test local LLM | EPIC-2 | 3 | Student 3 |
| **RAG-6** | Develop PDF text extractor | EPIC-3 | 3 | Student 1 |
| **RAG-7** | Develop text chunking module | EPIC-3 | 3 | Student 1 |
| **RAG-8** | Generate embeddings | EPIC-3 | 5 | Student 2 |
| **RAG-9** | Create vector database (ChromaDB) | EPIC-3 | 5 | Student 2 |
| **RAG-10**| Develop semantic retrieval | EPIC-3 | 5 | Student 2 |
| **RAG-11**| Integrate context with LLM | EPIC-3 | 5 | Student 3 |
| **RAG-12**| Develop RAG prompt | EPIC-3 | 3 | Student 3 |
| **RAG-13**| Implement source citation | EPIC-3 | 3 | Student 2 |
| **RAG-14**| Implement unknown-answer handling | EPIC-3 | 3 | Student 3 |
| **RAG-15**| Create Streamlit UI | EPIC-4 | 5 | Student 4 |
| **RAG-16**| Display retrieved sources & metadata | EPIC-4 | 3 | Student 4 |
| **RAG-17**| Perform functional testing | EPIC-5 | 5 | Student 4 |
| **RAG-18**| Evaluate LLM responses | EPIC-5 | 5 | Student 2 |
| **RAG-19**| Fix identified defects | EPIC-5 | 5 | Student 3 |
| **RAG-20**| Prepare documentation and demo | EPIC-6 | 5 | All Students |

---

## 🏛️ Directory Structure
```
Cloud-computing33/
├── app.py                     # Interactive Streamlit Web Application
├── document_loader.py         # PDF text extraction (PyPDF)
├── chunker.py                 # Text chunking with overlap
├── embeddings.py              # nomic-embed-text & fallback embeddings
├── vector_store.py            # ChromaDB local storage & retrieval
├── retriever.py               # Semantic query processor
├── llm_service.py             # Ollama llama3.2:3b interface & hallucination shield
├── rag_service.py             # End-to-end RAG orchestrator
├── generate_sample_pdfs.py    # Generates 5 realistic college PDF docs
├── data/                      # College PDF repository
│   ├── academic_regulations.pdf
│   ├── examination_guidelines.pdf
│   ├── student_handbook.pdf
│   ├── placement_guidelines.pdf
│   └── department_rules.pdf
├── chroma_db/                 # Persistent local ChromaDB database
├── tests/
│   ├── test_dataset.json      # 50 benchmark questions (Categories A, B, C)
│   ├── evaluate_rag.py        # Automated benchmark runner & metrics
│   └── test_rag.py            # Unit tests for components
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive documentation
```
