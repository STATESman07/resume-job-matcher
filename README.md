# resume-job-matcher
A Streamlit app that uses LLMs and embeddings to evaluate resumes against job descriptions and generate structured feedback with a fit score, strengths, weaknesses, and improvement suggestions.

# 📄 AI Resume Evaluator

A Streamlit-based web application that uses **Large Language Models (LLMs)** and **embeddings** to evaluate resumes against job descriptions. This tool helps job seekers understand how well their resume aligns with a specific job posting and provides actionable suggestions to improve it.

---

## 🚀 Features

- 📌 Upload your resume (PDF)
- 🧠 Paste a job description
- 🔍 Automatic resume parsing and chunking
- 🤖 LLM-powered evaluation using **Ollama**, **FAISS**, and **LangChain**
- 📝 Get a detailed report including:
  - ✅ Fit Score (0–100)
  - 💪 Key Strengths
  - ⚠️ Weaknesses
  - 🛠️ Suggestions for Improvement
  - 🚀 Top Skills
  - ❌ Missing Skills

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** [Ollama](https://ollama.com/) (`gemma3:1b`)
- **Embeddings:** `nomic-embed-text`
- **Vector Store:** FAISS
- **Framework:** LangChain
- **Document Loader:** PyMuPDF

---

## 📦 Installation

### 1. Clone the repository
git clone https://github.com/STATESman07/ai-resume-evaluator.git
cd ai-resume-evaluator


### 2. Create a virtual environment
conda create -n resume-eval python=3.10
conda activate resume-eval

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up environment variables
OPENAI_API_KEY=your_openai_key_if_needed
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=your_project_name

▶️ Run the App
streamlit run app.py



