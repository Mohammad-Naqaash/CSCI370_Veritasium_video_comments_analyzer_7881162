from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Load embedding model (must match what built the FAISS store) ──
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# ── Load saved FAISS vectorstore ───────────────────────────────────
vectorstore = FAISS.load_local(
    "./veritasium_faiss_store_2",
    embeddings,
    allow_dangerous_deserialization=True
)

# ── mmrRetrieval function (from 5_Retrieval.py) ────────────────────
def mmrRetrieval(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # mixes relevance with diversity
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.3  # 30% diversity, 70% relevance
        }
    )
    docs = retriever.invoke(question)
    text = []
    for doc in docs:
        text.append(doc.page_content)
    return text

# ── Test it ──────────────────────────────────────────────────────
query = "what do people think about gravity?"
results = mmrRetrieval(vectorstore, query)

print(f"Top results for: '{query}'\n")
for i, r in enumerate(results):
    print(f"{i+1}. {r}\n")