import streamlit as st
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Math PDF Assistant", page_icon="📐", layout="wide")

# ---------------- THE ULTIMATE CENTERED DOCK CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f0f4f9 !important; }
    
    .block-container { 
        max-width: 850px !important; 
        margin: 0 auto !important;
        padding-bottom: 280px !important; 
    }

    div[data-testid="stBottom"] > div { 
        background-color: transparent !important; 
        max-width: 850px !important;
        margin: 0 auto !important;
    }

    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px 16px 0 0;
        padding: 10px;
        margin-bottom: -1px !important;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.02);
    }

    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border-radius: 0 0 24px 24px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }

    .stChatMessageAvatar { display: none; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex; flex-direction: row-reverse; text-align: right;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        background-color: #e3e3e3; color: #1f1f1f; padding: 12px 20px; border-radius: 20px;
    }
    
    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- RAG LOGIC ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.warning("🔒 **API Token Required**")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def process_pdf_rag(file):
    try:
        reader = PdfReader(file)
        chunks = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                # Smaller chunks + bigger overlap = better math retrieval
                for i in range(0, len(content), 400):
                    chunks.append(content[i : i + 700])
        if not chunks: return None, None
        
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return chunks, index
    except: return None, None

def get_context(query, chunks, index, k=6): # Increased K to grab more quiz data
    query_vec = embedder.encode([query])
    _, indices = index.search(np.array(query_vec).astype('float32'), k)
    return "\n---\n".join([chunks[i] for i in indices[0]])

def generate_math_response(prompt, context):
    sys_prompt = (
        "You are an expert Math Assistant. Answer based ONLY on the PDF context.\n"
        "CRITICAL: You must provide a COMPLETE response for all items requested.\n"
        "FORMAT RULE: Wrap ALL mathematical expressions in double dollar signs ($$).\n"
        "Example: $$\\int x^2 dx = \\frac{x^3}{3} + C$$.\n\n"
        f"PDF CONTEXT:\n{context}"
    )
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    try:
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            # INCREASED LIMITS FOR 20 QUESTIONS
            max_tokens=4000, 
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Service Error: {str(e)}"

# ---------------- APP STATE ----------------
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "chunks" not in st.session_state: st.session_state.chunks = None

# ---------------- UI: MESSAGES ----------------
if not st.session_state.messages:
    st.markdown("<h1 style='text-align:center; margin-top:10vh; color:#1f1f1f;'>Math PDF Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Upload your quiz. The document is purged after each response.</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- UI: CENTERED DOCK ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed", key="dock_upload")

if uploaded_file:
    with st.spinner(" "):
        st.session_state.chunks, st.session_state.vector_db = process_pdf_rag(uploaded_file)

user_query = st.chat_input("Analyze document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.vector_db is not None:
            with st.spinner("Solving all items..."):
                # Grab more context (K=6) to ensure all 20 questions are found
                context = get_context(user_query, st.session_state.chunks, st.session_state.vector_db, k=6)
                ans = generate_math_response(user_query, context)
                
                # Cleanup common AI formatting mistakes
                ans = ans.replace(r"\(", "$$").replace(r"\)", "$$")
                ans = ans.replace(r"\[", "$$").replace(r"\]", "$$")
        else:
            ans = "❌ **Document Required.** Please upload a PDF."
        
        st.markdown(ans)
    
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.session_state.vector_db = None 
    st.session_state.chunks = None
    st.rerun()