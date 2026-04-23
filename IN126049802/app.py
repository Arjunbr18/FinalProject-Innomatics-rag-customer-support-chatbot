import os
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.graph_flow import build_graph

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("Customer Support Assistant (RAG + HITL)")


if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "hitl_active" not in st.session_state:
    st.session_state.hitl_active = False

if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = ""


# ---- Upload PDF ----
uploaded_file = st.file_uploader("Upload PDF", type="pdf")


# ---- Process PDF ----
if uploaded_file and not st.session_state.db_ready:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded")

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    st.session_state.db_ready = True
    st.success("PDF Ready for Questions")


# ---- Chat Section ----
if st.session_state.db_ready:

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---- NORMAL CHAT ----
    if not st.session_state.hitl_active:

        if prompt := st.chat_input("Ask your question..."):

            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.write(prompt)

            app = build_graph()

            result = app.invoke({
                "query": prompt,
                "answer": ""
            })

            # ---- HITL TRIGGER ----
            if result["answer"] == "HITL_REQUIRED":

                st.session_state.hitl_active = True

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No answer found. Human input required."
                })

                st.rerun()

            else:
                response = result["answer"]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                with st.chat_message("assistant"):
                    st.write(response)

    # ---- HITL MODE ----
    else:
        with st.chat_message("assistant"):
            st.warning("Please enter human response:")

        human_input = st.text_input("Human Response", key="hitl_box")

        if human_input and human_input != st.session_state.last_submitted:

            st.session_state.last_submitted = human_input

            st.session_state.messages.append({
                "role": "assistant",
                "content": human_input
            })

            st.session_state.hitl_active = False

            st.rerun()

        # ---- Buttons ----
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Submit Response"):
                if human_input.strip() != "":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": human_input
                    })

                    st.session_state.hitl_active = False
                    st.rerun()

        with col2:
            if st.button("Skip"):
                st.session_state.hitl_active = False
                st.rerun()
