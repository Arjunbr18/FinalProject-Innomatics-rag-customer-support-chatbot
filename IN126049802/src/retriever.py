import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

# Load once
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

def retrieve_docs(query):
    return db.similarity_search(query, k=3)

def generate_answer(query, docs):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a customer support assistant.
Answer ONLY using the context.
If not found, say "I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = input("Enter your question: ")

    docs = retrieve_docs(query)
    answer = generate_answer(query, docs)

    print("\nAnswer:")
    print(answer)