# RAG-Based Customer Support Chatbot

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot designed to answer user queries using custom documents.

Instead of depending only on pre-trained knowledge, the system retrieves relevant information from stored documents and generates accurate, context-aware responses. This improves reliability and makes the chatbot suitable for customer support use cases.

---

## Features

- Document-based Question Answering  
- Semantic Search using Embeddings  
- Context-aware Response Generation  
- Fast Retrieval using Vector Database (ChromaDB)  
- Simple and Interactive UI using Streamlit  

---

## Tech Stack

- Python  
- LangChain  
- OpenAI API (LLM)  
- ChromaDB (Vector Database)  
- Streamlit (Frontend UI)  

---

## Project Structure

```
RAG_Project/
│
├── app.py                  # Main application (UI + execution)
├── requirements.txt        # Dependencies
│
├── src/
│   ├── ingest.py           # Document loading & chunking
│   ├── retriever.py        # Retrieval logic
│   ├── graph_flow.py       # RAG pipeline workflow
│

```

---

## How It Works

1. Load documents from the `data/` folder  
2. Split documents into smaller chunks  
3. Convert chunks into embeddings  
4. Store embeddings in ChromaDB  
5. Retrieve relevant chunks based on user query  
6. Generate final answer using LLM  

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-customer-support-chatbot.git
cd rag-customer-support-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create a `.env` file in the root folder and add:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Demo

(https://drive.google.com/file/d/16rdbY-8GetcgTs9I27txT4FxDgX9sfup/view?usp=sharing)

---

## Future Improvements

- Improve retrieval accuracy  
- Enhance UI for better user experience  
- Add real-time document upload feature  

---

## Author

**Arjun BR**
