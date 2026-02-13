import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Smart Email Writer",
    page_icon="📧",
    layout="wide"
)

st.sidebar.title("📧 Smart Email Writer")
page = st.sidebar.radio(
    "Navigation",
    ["Generate Email", "Improve Email", "About"]
)


# ---------- VECTOR DB SETUP ----------
if not os.path.exists("chroma_db"):
    os.makedirs("chroma_db", exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("email_templates")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_context(query):
    embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )

    if results["documents"]:
        return "\n".join(results["documents"][0])
    return ""


# ---------- LOAD API KEY ----------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key missing")
    st.stop()

client = genai.Client(api_key=api_key)


# =====================================================
# GENERATE EMAIL PAGE
# =====================================================

if page == "Generate Email":

    st.title("✉ Generate Smart Email")

    purpose = st.text_input("Email Purpose")

    # STEP-3 UI Columns
    col1, col2 = st.columns(2)

    with col1:
        tone = st.selectbox(
            "Tone",
            ["Professional", "Friendly", "Formal", "Apologetic", "Persuasive"]
        )

    with col2:
        length = st.selectbox(
            "Email Length",
            ["Short", "Medium", "Detailed"]
        )

    col3, col4 = st.columns(2)

    with col3:
        generate_subject = st.checkbox("Auto Generate Subject Line")

    with col4:
        improve_mode = st.checkbox("Improve Existing Email")

    existing_email = ""
    if improve_mode:
        existing_email = st.text_area("Paste your email here")

    if st.button("Generate Email"):

        if not purpose and not improve_mode:
            st.warning("Enter email purpose")
            st.stop()

        context = retrieve_context(purpose)

        # Subject generator
        subject_line = ""
        if generate_subject and purpose:
            subject_prompt = f"Generate professional subject line for: {purpose}"

            subject_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=subject_prompt
            )
            subject_line = subject_response.text

        # Prompt creation
        if improve_mode and existing_email:
            prompt = f"""
Improve this email professionally:

{existing_email}

Fix grammar, tone, clarity and professionalism.
"""
        else:
            prompt = f"""
Use this context:

{context}

Subject suggestion:
{subject_line}

Write a {length.lower()} {tone.lower()} professional email about:

{purpose}

Include greeting, body and closing.
"""

        try:
            with st.spinner("Generating email..."):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

            if subject_line:
                st.write("### Suggested Subject:")
                st.write(subject_line)

            st.subheader("Generated Email")

            st.text_area(
                "Email Output",
                response.text,
                height=300
            )

            st.download_button(
                "Copy / Download Email",
                response.text,
                file_name="generated_email.txt"
            )

        except Exception as e:
            st.error(f"Generation error: {e}")


# =====================================================
# IMPROVE EMAIL PAGE
# =====================================================

elif page == "Improve Email":

    st.title("📝 Improve Existing Email")

    existing_email = st.text_area("Paste your email", height=300)

    if st.button("Improve Email"):

        prompt = f"""
Improve this email professionally:

{existing_email}

Fix grammar, tone, clarity and formatting.
"""

        with st.spinner("Improving email..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.text_area("Improved Email", response.text, height=300)


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("📘 About This App")

    st.write("""
This Smart Email Writer uses:

✅ Google Gemini AI  
✅ Vector RAG (ChromaDB)  
✅ Sentence Transformers  
✅ Streamlit UI  

Features:

• Generate professional emails  
• Improve existing emails  
• Auto subject generation  
• Tone and length control  
• Copy/download email option  

Built as a GenAI portfolio project.
""")
