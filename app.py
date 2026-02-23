import os
import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

DATA_FOLDER = "data"
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="RAGenius-ChatBot", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0F172A;
    color: white;
}
h1 {
    color: #38BDF8;
}
.response-box {
    background-color: #334155;
    padding: 20px;
    border-radius: 12px;
    height: 300px;
    overflow: auto;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 RAGenius-ChatBot")

import os
from groq import Groq

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def call_llm(prompt):
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


# -------------------------
# File Loaders
# -------------------------

def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def load_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_ppt(file_path):
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_all_documents():
    documents = []

    if not os.path.exists(DATA_FOLDER):
        return documents

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        try:
            if file.endswith(".pdf"):
                text = load_pdf(path)
            elif file.endswith(".docx"):
                text = load_docx(path)
            elif file.endswith(".pptx"):
                text = load_ppt(path)
            elif file.endswith(".txt"):
                text = load_txt(path)
            else:
                continue

            documents.append(Document(page_content=text))
        except:
            continue

    return documents


# -------------------------
# Vector Store Creation
# -------------------------

def create_vector_store():
    docs = load_all_documents()

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# # Initialize session state
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = create_vector_store()

if "response" not in st.session_state:
    st.session_state.response = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -------------------------
# RAG Query Function
# -------------------------

def unified_query(question):

    vector_store = st.session_state.vector_store
    question_lower = question.lower()

    # Detect type of answer
    if "brief" in question_lower:
        length_instruction = "Give a brief answer in 1-2 short sentences."
    elif "explain" in question_lower or "detailed" in question_lower:
        length_instruction = "Provide a clear and detailed explanation in 5-6 sentences."
    else:
        length_instruction = "Answer in exactly 3 complete sentences."

    # 🟢 No file uploaded → GK mode
    if vector_store is None:
        prompt = f"""
You are an intelligent AI assistant.

{length_instruction}

Question:
{question}

Answer:
"""
        return call_llm(prompt)

    # 🟢 File uploaded → RAG mode
    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return "No details found in the uploaded documents."

    context = "\n".join([doc.page_content for doc in docs]).strip()

    if not context:
        return "No details found in the uploaded documents."

    prompt = f"""
You are a RAG assistant.

{length_instruction}

Answer ONLY using the provided context.
If the answer is not clearly present, respond exactly with:

No details found in the uploaded documents.

Context:
{context}

Question:
{question}

Answer:
"""

    return call_llm(prompt)

# -------------------------
# UI
# -------------------------

st.markdown("### 🤖 Response")

st.markdown(
    f"""
    <div class="response-box">
    {st.session_state.response}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col1:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("💬 Ask your question")
        submit = st.form_submit_button("Send")

with col2:
    uploaded_file = st.file_uploader(
        "📂 Upload",
        type=["pdf", "docx", "pptx", "txt"]
    )


# -------------------------
# File Upload Handling
# -------------------------

if uploaded_file is not None:
    os.makedirs(DATA_FOLDER, exist_ok=True)

    save_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # Rebuild vector store immediately
    st.session_state.vector_store = create_vector_store()
    st.success("Knowledge base updated!")


# -------------------------
# Question Handling
# -------------------------

if submit and question:
    with st.spinner("Generating response..."):
        answer = unified_query(question)
        st.session_state.response = answer
        st.rerun()