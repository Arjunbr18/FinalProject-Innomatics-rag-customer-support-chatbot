import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

# Step 2: Split text
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

# Step 3: Embeddings
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# Step 4: Store in Chroma
def store_in_chroma(chunks, embeddings):
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    db.persist()
    print("Stored in ChromaDB")
    return db


if __name__ == "__main__":
    docs = load_pdf("data/sample.pdf")
    chunks = split_text(docs)
    embeddings = create_embeddings()
    store_in_chroma(chunks, embeddings)

    print("Ingestion complete")