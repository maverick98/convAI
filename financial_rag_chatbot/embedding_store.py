import os
import PyPDF2
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# Define directories
DATA_FOLDER = "data/"
VECTOR_DB_FOLDER = "vector_db/"
CHROMADB_PATH = os.path.join(VECTOR_DB_FOLDER, "chromadb_store")

# Load the Sentence Transformer model for embedding generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
#chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
chroma_client = chromadb.PersistentClient(path="./vector_db/chromadb_store")


collection = chroma_client.get_or_create_collection(name="financial_docs")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and splits it into smaller text chunks for better retrieval.
    Uses sentence tokenization and groups sentences into 3-sentence chunks.
    """
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Tokenize text into sentences and group into 3-sentence chunks
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    return chunks

def load_financial_data():
    """
    Loads financial data from all PDFs in the 'data/' folder.
    Extracts text and converts it into chunks for better retrieval.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_FOLDER}'. Please add financial reports.")

    all_chunks = []
    for pdf in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf)
        print(f"📄 Extracting text from {pdf_path}...")
        text_chunks = extract_text_from_pdf(pdf_path)
        all_chunks.extend(text_chunks)

    return all_chunks  # Returns a list of text chunks

def store_embeddings():
    """
    Generates embeddings for all financial document chunks and stores them in ChromaDB.
    """
    os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

    text_chunks = load_financial_data()
    print("🔄 Generating embeddings...")

    embeddings = embedding_model.encode(text_chunks).tolist()

    # Store in ChromaDB
    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
        collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"text": text}])

    print(f"✅ Stored {len(embeddings)} embeddings in ChromaDB at {CHROMADB_PATH}")

if __name__ == "__main__":
    store_embeddings()
