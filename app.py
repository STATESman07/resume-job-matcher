import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# LangChain components
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI setup
st.set_page_config(page_title="LLM Resume Evaluator", layout="wide")
st.title("📄 AI Resume Evaluator using LLM + Embeddings")

# UI inputs
resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here", height=250)
evaluate_button = st.button("🔍 Evaluate Resume")

if evaluate_button:
    if not resume_file:
        st.warning("Please upload a valid PDF resume.")
        st.stop()
    if not job_description.strip():
        st.warning("Please paste the job description.")
        st.stop()

    try:
        with st.spinner("Processing..."):
            # Save resume temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resume_file.read())
                resume_path = tmp_file.name

            # Load resume content
            loader = PyMuPDFLoader(resume_path)
            documents = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)

            # Embedding and Vector Store
            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
            retriever = vectorstore.as_retriever()

            # Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                """You are an expert resume evaluator. Based on the resume and the job description provided, analyze and return a structured evaluation report with the following sections:

1. Fit Score: Return a numerical value from 0 to 100.
2. Strengths: Bullet points of the candidate's key strengths relevant to the job.
3. Weaknesses: Bullet points describing any potential gaps or concerns.
4. Suggestions: Bullet points suggesting actionable improvements to enhance the resume or qualifications.
5. Top Skills: List of most relevant skills the candidate already possesses.
6. Missing Skills: Important job-required skills that are not evident in the resume.

Formatting Guidelines:
- Format each section with a clear header followed by bullet points.
- Do NOT repeat the section title inside bullet points.
- Do NOT include any extra explanations or comments outside the defined sections.
- Do NOT output anything except the report in this format.

Resume:
{context}

Job Description:
""" + job_description),
                ("human", "{input}"),
            ])

            # LLM and Retrieval Chain
            llm = OllamaLLM(model="gemma3:1b")
            doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)

            # Final query
            query = "Evaluate the resume based on the job description above."
            response = retrieval_chain.invoke({"input": query})
            full_output = response["answer"]

            # Extract and display Fit Score
            fit_score_match = re.search(r"\*\*1\. Fit Score:\*\*\s*(\d+)", full_output)
            if fit_score_match:
                fit_score = int(fit_score_match.group(1))
                fit_score = max(0, min(100, fit_score))  # Clamp value
                st.markdown(f"## 🔢 Fit Score: `{fit_score}%`")
                st.progress(fit_score / 100.0)

            st.success("✅ Evaluation Complete")
            st.subheader("📝 Evaluation Report")

            # Section titles to icons
            section_icons = {
                "Strengths": "💪",
                "Weaknesses": "⚠️",
                "Suggestions": "🛠️",
                "Top Skills": "🚀",
                "Missing Skills": "❌",
            }

            # Parse sections
            sections = {}
            section_pattern = re.compile(r"\*\*\d+\.\s*(.*?)\s*:\*\*")
            matches = list(section_pattern.finditer(full_output))

            for idx, match in enumerate(matches):
                section_name = match.group(1).strip()
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_output)
                content_block = full_output[start:end].strip()

                # Extract bullet points
                bullets = [
                    line.strip(" *").strip()
                    for line in content_block.splitlines()
                    if line.strip().startswith("*")
                ]
                sections[section_name] = bullets

            # Render sections
            any_displayed = False
            for title, icon in section_icons.items():
                if title in sections:
                    st.markdown(f"### {icon} {title}")
                    for bullet in sections[title]:
                        st.markdown(f"- {bullet}")
                    any_displayed = True

            # Fallback
            if not any_displayed:
                st.warning("⚠️ Could not extract structured sections. Showing raw output.")
                st.code(full_output)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

    finally:
        if "resume_path" in locals() and os.path.exists(resume_path):
            os.remove(resume_path)
