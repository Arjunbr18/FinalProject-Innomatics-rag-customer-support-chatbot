import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END
from groq import Groq


# ---- STATE ----
class GraphState(TypedDict):
    query: str
    context: str
    answer: str


# ---- INIT MODELS ----
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---- PROCESS NODE ----
def process_query(state: GraphState):

    query = state["query"]

    docs = db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=10
    )

    if not docs:
        return {"context": "", "answer": "NOT FOUND"}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a customer support assistant.

Answer strictly from the given context.

If the question is broad:
- Try to summarize relevant points from context.

If NO relevant information is found:
Return: NOT FOUND

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

    answer = response.choices[0].message.content.strip()

    return {
        "context": context,
        "answer": answer
    }


# ---- ROUTER ----
def route(state: GraphState):
    if state["answer"] == "NOT FOUND":
        return "hitl"
    return "end"


# ---- HITL NODE ----
def hitl_node(state: GraphState):
    return {"answer": "HITL_REQUIRED"}


# ---- BUILD GRAPH ----
def build_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node("process", process_query)
    workflow.add_node("hitl", hitl_node)

    workflow.set_entry_point("process")

    workflow.add_conditional_edges(
        "process",
        route,
        {
            "hitl": "hitl",
            "end": END
        }
    )

    workflow.add_edge("hitl", END)

    return workflow.compile()
