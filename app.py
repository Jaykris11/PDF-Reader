import streamlit as st
from pypdf import PdfReader
from huggingface_hub import InferenceClient
import os
import time
from dotenv import load_dotenv

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Math PDF Assistant", page_icon="📐", layout="wide")

# ---------------- MODERN UI CSS ----------------
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

    /* CENTERED FLOATING BOTTOM DOCK */
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

# ---------------- LOGIC & SECURITY ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Portfolio Fix: Prevent crash if recruiter hasn't set the token
if not HF_TOKEN:
    st.warning("🔒 **API Token Required**: To run this assistant, please set your `HF_TOKEN` in environment variables or Streamlit secrets.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

def load_pdf(file):
    try:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        if not text.strip():
            return "EMPTY_ERR"
        return text
    except Exception:
        return "LOAD_ERR"

def generate(prompt, context=""):
    sys_prompt = (
        "You are an expert Math Document Assistant. "
        "Strictly analyze the PDF and solve mathematical problems. "
        "Format all math using LaTeX with double dollar signs: $$...$$. "
        "Refuse to answer questions unrelated to the document content.\n\n"
        f"PDF TEXT:\n{context[:4000]}"
    )
    
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    try:
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Service currently unavailable: {str(e)}"

# ---------------- APP STATE ----------------
if "messages" not in st.session_state: st.session_state.messages = []
if "temp_pdf" not in st.session_state: st.session_state.temp_pdf = ""

# ---------------- MAIN VIEW ----------------
if not st.session_state.messages:
    st.markdown("<h1 style='text-align:center; margin-top:10vh; color:#1f1f1f;'>Math PDF Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Upload your quiz below to begin. Documents are purged after each response.</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CENTERED DOCK ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed", key="dock_upload")
if uploaded_file:
    with st.spinner(" "):
        st.session_state.temp_pdf = load_pdf(uploaded_file)

user_query = st.chat_input("Analyze document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.temp_pdf:
            ans = "❌ **Document Required.** Please upload a PDF in the dock for every prompt."
        elif st.session_state.temp_pdf == "EMPTY_ERR":
            ans = "⚠️ **Scanned Image Detected.** This PDF contains no selectable text. Please use a text-based PDF."
        elif st.session_state.temp_pdf == "LOAD_ERR":
            ans = "❌ **Load Error.** Could not read the file. Try a different PDF."
        else:
            with st.spinner(" "):
                ans = generate(user_query, st.session_state.temp_pdf)
        
        st.markdown(ans)
    
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.session_state.temp_pdf = "" # The Purge
    st.rerun()