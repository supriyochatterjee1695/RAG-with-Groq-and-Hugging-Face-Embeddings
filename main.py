import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

class CSeametryBot:
    def __init__(self):
        self.db_path = "faiss_index"
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        if not self.groq_key:
            raise ValueError("❌ GROQ_API_KEY missing from .env")

        # 1. Load Embeddings Model (Same as used in ingestion)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. Load LLM
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=self.groq_key, 
            model_name=self.model_name
        )
        
        self.vector_db = self._load_db()

    def _load_db(self):
        """Loads the pre-built FAISS index"""
        if os.path.exists(self.db_path):
            print("📂 Knowledge base loaded successfully.")
            return FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("⚠️ No knowledge base found. Please run ingest.py first.")
            return None

    def ask(self, query: str):
        if not self.vector_db:
            return "Knowledge base is empty."

        # Production-grade retrieval (Top 3 chunks)
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain.invoke({"query": query})

if __name__ == "__main__":
    bot = CSeametryBot()
    
    print("\n--- CSeametry AI Consultant Active ---")
    while True:
        user_input = input("\n❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = bot.ask(user_input)
        
        print(f"\n🤖 AI: {response['result']}")
        print("\n📍 Sources used:")
        for doc in response['source_documents']:
            page = doc.metadata.get('page', 'N/A')
            print(f"- [Page {page}]: {doc.page_content[:80]}...")