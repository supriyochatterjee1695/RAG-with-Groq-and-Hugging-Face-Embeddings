import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion(file_path: str):
    embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    db_path = "faiss_index"

    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found.")
        return

    print(f"📑 Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Semantic Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(documents)

    # Vectorization
    print(f"🧬 Generating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create and Save Index
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(db_path)
    print(f"💾 Vector database successfully saved to '{db_path}'.")

if __name__ == "__main__":
    # You can change this to any PDF you want to process
    run_ingestion("CSeametry.pdf")