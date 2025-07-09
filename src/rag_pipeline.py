"""
rag_pipeline.py

Task 3: RAG Core Logic and Evaluation
- Retriever: retrieves top-k relevant complaint chunks for a user question
- Prompt engineering: robust template for LLM
- Generator: combines context and question, sends to LLM, returns answer
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List

# Parameters
VECTOR_STORE_DIR = "../vector_store"
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
K = 5  # top-k retrieved chunks

PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, state that you don't have enough information.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

# Load embedding model and vector store
model = SentenceTransformer(f"sentence-transformers/{EMBEDDING_MODEL}")
client = chromadb.Client(Settings(persist_directory=VECTOR_STORE_DIR))
collection = client.get_or_create_collection("complaints")

def retrieve_context(question: str, k: int = K) -> List[str]:
    """Embed the question and retrieve top-k relevant complaint chunks."""
    q_emb = model.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(docs, metas))

def build_prompt(context_chunks: List[tuple], question: str) -> str:
    context = "\n---\n".join([f"[{meta.get('product', '')}] {doc}" for doc, meta in context_chunks])
    return PROMPT_TEMPLATE.format(context=context, question=question)

# Generator: using Hugging Face pipeline (local LLM) or OpenAI (if available)
def generate_answer(prompt: str) -> str:
    try:
        from transformers import pipeline
        # Use a small open-source LLM for demo (e.g., distilgpt2)
        generator = pipeline("text-generation", model="distilgpt2")
        output = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
        return output[0]["generated_text"][len(prompt):].strip()
    except ImportError:
        return "[ERROR] transformers not installed. Please install with 'pip install transformers'."
    except Exception as e:
        return f"[ERROR] {e}"

def rag_answer(question: str, k: int = K) -> dict:
    context_chunks = retrieve_context(question, k)
    prompt = build_prompt(context_chunks, question)
    answer = generate_answer(prompt)
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": context_chunks[:2]  # show 1-2 for evaluation
    }

if __name__ == "__main__":
    # Example usage
    questions = [
        "Why are people unhappy with BNPL?",
        "What are the most common complaints about credit cards?",
        "Are there issues with money transfers?",
        "How do customers feel about personal loans?",
        "What problems do users report with savings accounts?"
    ]
    for q in questions:
        result = rag_answer(q)
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("Retrieved Sources:")
        for doc, meta in result['retrieved_sources']:
            print(f"- [{meta.get('product', '')}] {doc}")