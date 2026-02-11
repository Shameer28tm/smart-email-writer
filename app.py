import streamlit as st
import os
from dotenv import load_dotenv
from google import genai


# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key missing in .env file")
    st.stop()


# Configure client
client = genai.Client(api_key=api_key)


# UI
st.set_page_config(page_title="Smart Email Writer")
st.title("📧 Smart Email Writer (GenAI)")


purpose = st.text_input("Email Purpose")

tone = st.selectbox(
    "Tone",
    ["Professional", "Friendly", "Formal", "Apologetic", "Persuasive"]
)


if st.button("Generate Email"):

    if not purpose:
        st.warning("Enter email purpose first.")
        st.stop()

    prompt = f"""
Write a {tone.lower()} professional email about:

{purpose}

Include:
- Subject line
- Greeting
- Clear professional body
- Closing signature
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.subheader("Generated Email")
        st.write(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
