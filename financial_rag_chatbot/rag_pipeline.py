import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from embedding_store import load_financial_data, FAISS_INDEX_PATH

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def load_faiss_index():
    """
    Loads the FAISS index from disk.
    """
    return faiss.read_index(FAISS_INDEX_PATH)

financial_documents = load_financial_data()
tokenized_corpus = [doc.split() for doc in financial_documents]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_faiss(query, top_k=15):
    """
    Retrieves the most relevant financial statements using FAISS.
    """
    index = load_faiss_index()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    return [financial_documents[i] for i in indices[0] if i < len(financial_documents)]

def retrieve_bm25(query, top_k=10):
    """
    Retrieves the most relevant financial statements using BM25 (Keyword Search).
    """
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [financial_documents[i] for i in top_indices]

def extract_financial_data(text):
    """
    Extracts key financial figures from retrieved text.
    """

    finance_terms = {
        "net_income": r"(Net (income|loss).*?\$?(-?\d+[\d,.]*)\s?(billion|million)?)",
        "revenue": r"(Revenue[s]?.*?\$?(-?\d+[\d,.]*)\s?(billion|million)?)",
        "total_assets": r"(Total assets.*?\$?(-?\d+[\d,.]*)\s?(billion|million)?)",
        "debt": r"(Debt.*?\$?(-?\d+[\d,.]*)\s?(billion|million)?)",
        "free_cash_flow": r"(Free cash flow.*?\$?(-?\d+[\d,.]*)\s?(billion|million)?)"
    }

    for key, pattern in finance_terms.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}"

    financial_numbers = re.findall(r"\$?-?\d+[\d,.]*\s?(billion|million)?", text, re.IGNORECASE)
    
    return " ".join(financial_numbers[:3]) if financial_numbers else "No financial data found."

def re_rank_results(query, retrieved_docs):
    """
    Re-ranks retrieved documents using a cross-encoder model.
    Returns the best answer along with a confidence score.
    """
    if not retrieved_docs:
        return "No relevant document found.", 0.0

    query_pairs = [(query, doc) for doc in retrieved_docs]
    scores = cross_encoder.predict(query_pairs)

    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    best_answer = extract_financial_data(ranked_results[0][0])
    confidence = ranked_results[0][1]  # Assign confidence score

    return best_answer, confidence

def retrieve_answer(query):
    """
    Executes the full retrieval pipeline: FAISS + BM25 + Cross-Encoder Re-Ranking.
    Returns answer + confidence score.
    """
    faiss_results = retrieve_faiss(query)
    bm25_results = retrieve_bm25(query)
    combined_results = list(set(faiss_results + bm25_results))

    return re_rank_results(query, combined_results)
