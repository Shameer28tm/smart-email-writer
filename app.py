import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer


# ---------- Vector DB Setup ----------
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


# ---------- Load API Key ----------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key missing in .env file")
    st.stop()

client = genai.Client(api_key=api_key)


# ---------- UI ----------
st.set_page_config(page_title="Smart Email Writer")
st.title("📧 Smart Email Writer (GenAI + Vector RAG)")


purpose = st.text_input("Email Purpose")

tone = st.selectbox(
    "Tone",
    ["Professional", "Friendly", "Formal", "Apologetic", "Persuasive"]
)

generate_subject = st.checkbox("Auto Generate Subject Line")

length = st.selectbox(
    "Email Length",
    ["Short", "Medium", "Detailed"]
)

improve_mode = st.checkbox("Improve Existing Email")

existing_email = ""
if improve_mode:
    existing_email = st.text_area("Paste your email here")


# ---------- Generate Button ----------
if st.button("Generate Email"):

    if not purpose and not improve_mode:
        st.warning("Enter email purpose first.")
        st.stop()

    # Retrieve RAG context
    context = retrieve_context(purpose)

    # Generate subject line if needed
    subject_line = ""
    if generate_subject and purpose:
        subject_prompt = f"Generate a professional email subject line for: {purpose}"
        subject_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=subject_prompt
        )
        subject_line = subject_response.text

    # Prompt logic
    if improve_mode and existing_email:
        prompt = f"""
Improve this email professionally:

{existing_email}

Fix:
- Grammar
- Tone
- Clarity
- Professional formatting
"""
    else:
        prompt = f"""
Use this context:

{context}

Subject suggestion:
{subject_line}

Write a {length.lower()} {tone.lower()} professional email about:
{purpose}

Include:
- Proper greeting
- Clear professional body
- Closing signature
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if subject_line:
            st.write("### Suggested Subject:")
            st.write(subject_line)

        st.subheader("Generated Email")
        st.write(response.text)

        st.download_button(
            "Copy / Download Email",
            response.text,
            file_name="generated_email.txt"
        )

    except Exception as e:
        st.error(f"Generation error: {e}")
