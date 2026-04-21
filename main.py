import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

class ProductionRAG:
    def __init__(self):
        # Fetching from .env
        groq_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        
        if not groq_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")

        # 1. Initialize LLM
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=groq_key, 
            model_name=model_name
        )
        
        # 2. Initialize Embeddings (Runs locally via HuggingFace transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_db = None
        self.db_path = "faiss_index"

    def ingest_docs(self, file_path: str):
        """Processes PDF into searchable chunks"""
        if not os.path.exists(file_path):
            print(f"❌ Error: File {file_path} not found.")
            return

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"🧬 Embedding {len(chunks)} chunks...")
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(self.db_path)
        print("💾 Vector DB saved locally.")

    def load_db(self):
        """Loads the local FAISS index"""
        if os.path.exists(self.db_path):
            self.vector_db = FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
        return False

    def ask(self, question: str):
        """Retrieval + Generation"""
        if not self.vector_db:
            if not self.load_db():
                return "No data found. Please ingest a document first."

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Custom Prompt to ensure the AI uses your data
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain.invoke({"query": question})

# --- RUNTIME ---
if __name__ == "__main__":
    bot = ProductionRAG()
    
    # 1. Ingest (uncomment for the first run)
    bot.ingest_docs("CSeametry.pdf")
    
    # 2. Chat Loop
    print("\n--- RAG Tool Ready (Type 'exit' to quit) ---")
    while True:
        query = input("\n❓ Question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        response = bot.ask(query)
        
        print(f"\n🤖 AI: {response['result']}")
        print("\n📍 Sources:")
        for doc in response['source_documents']:
            print(f"- [Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content[:70]}...")