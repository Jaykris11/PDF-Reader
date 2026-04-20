import streamlit as st
from pypdf import PdfReader
from huggingface_hub import InferenceClient
import os
import time
from dotenv import load_dotenv

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF Reader", page_icon="📐", layout="wide")

# ---------------- THE ULTIMATE CENTERED DOCK CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f0f4f9 !important; }
    
    /* Center the main content area */
    .block-container { 
        max-width: 850px !important; 
        margin: 0 auto !important;
        padding-bottom: 280px !important; 
    }

    /* --------------------------------------
       THE FIX: CENTERED FLOATING BOTTOM DOCK
       -------------------------------------- */
    div[data-testid="stBottom"] {
        background-color: transparent !important;
    }

    /* Forces the bottom container to match the width of your messages */
    div[data-testid="stBottom"] > div { 
        background-color: transparent !important; 
        max-width: 850px !important;
        margin: 0 auto !important;
        padding: 0px 20px !important;
    }

    /* Integrated Uploader Box */
    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px 16px 0 0;
        padding: 10px;
        margin-bottom: -1px !important;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.02);
    }

    /* Pill Chat Input - FORCED ALIGNMENT */
    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border-radius: 0 0 24px 24px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }

    /* Chat Bubbles Styling */
    .stChatMessageAvatar { display: none; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex; flex-direction: row-reverse; text-align: right; background-color: transparent;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        background-color: #e3e3e3; color: #1f1f1f; padding: 12px 20px; border-radius: 20px;
    }
    
    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIC ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

def load_pdf(file):
    try:
        reader = PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except: return ""

def generate(prompt, context=""):
    sys_prompt = (
        "You are an expert Math Document Assistant. "
        "CRITICAL: When writing math, use double dollar signs for equations like this: $$integral symbols$$. "
        "Do not use single slashes or parentheses for math. "
        "ONLY answer questions based on the uploaded PDF. "
        "If the user asks something unrelated to the document, refuse to answer.\n\n"
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
        return f"Error: {str(e)}"

# ---------------- STATE ----------------
if "messages" not in st.session_state: st.session_state.messages = []
if "temp_pdf" not in st.session_state: st.session_state.temp_pdf = ""

# ---------------- UI ----------------
if not st.session_state.messages:
    st.markdown("<h1 style='text-align:center; margin-top:10vh; color:#1f1f1f;'> PDF Reader</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Upload your quiz and ask a question. The document is cleared after every response.</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- THE DOCK (Always centered) ---
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed", key="dock_upload")
if uploaded_file:
    st.session_state.temp_pdf = load_pdf(uploaded_file)

user_query = st.chat_input("Ask about the PDF...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.temp_pdf:
            ans = "❌ **No PDF found.** Please upload a file in the dock above for each prompt."
            st.markdown(ans)
        else:
            with st.spinner("Processing..."):
                ans = generate(user_query, st.session_state.temp_pdf)
            st.markdown(ans)
    
    st.session_state.messages.append({"role": "assistant", "content": ans})
    
    # Reset PDF state so it must be uploaded again for next prompt
    st.session_state.temp_pdf = "" 
    st.rerun()