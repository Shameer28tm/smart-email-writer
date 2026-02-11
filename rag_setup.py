import chromadb
from sentence_transformers import SentenceTransformer

# Persistent DB folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("email_templates")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Read templates
with open("data/email_templates.txt", "r") as f:
    docs = f.readlines()

for i, doc in enumerate(docs):
    embedding = embed_model.encode(doc).tolist()

    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("✅ Vector DB created successfully!")
