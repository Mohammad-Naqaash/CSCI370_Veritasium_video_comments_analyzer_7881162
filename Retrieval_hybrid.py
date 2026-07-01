import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Load comments ─────────────────────────────────────────────
df = pd.read_csv("Processed comments/Veritasium_comments_modeling_topics_3.0.csv")
df = df.dropna(subset=["text_normalized"]).reset_index(drop=True)

# ── Lexical setup (TF-IDF) ───────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['tokens_str'].fillna(''))

# ── Semantic setup (FAISS) ───────────────────────────────────────
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
vectorstore = FAISS.load_local("./veritasium_faiss_store_2", embeddings, allow_dangerous_deserialization=True)

# ── Hybrid retrieval function (efficient version) ────────────────
def hybridRetrieval(query, top_n=5, pool_size=100, alpha=0.5):
    # Step 1: get top semantic candidates only (not the whole dataset)
    semantic_results = vectorstore.similarity_search_with_score(query, k=pool_size)
    semantic_scores = {doc.page_content: score for doc, score in semantic_results}

    # Step 2: get lexical scores only for those same candidates
    candidate_texts = list(semantic_scores.keys())
    candidate_rows = df[df['text_normalized'].isin(candidate_texts)]

    query_vec = tfidf.transform([query])

    combined = []
    for idx, row in candidate_rows.iterrows():
        text = row['text_normalized']
        lex_score = cosine_similarity(query_vec, tfidf_matrix[idx]).flatten()[0]
        sem_score = 1 / (1 + semantic_scores.get(text, 999))
        final_score = alpha * sem_score + (1 - alpha) * lex_score
        combined.append((text, row['like_count'], final_score))

    combined.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(combined[:top_n], columns=['text_normalized', 'like_count', 'hybrid_score'])

# ── Test it ──────────────────────────────────────────────────────
query = "force gravity mass"
results = hybridRetrieval(query, top_n=5)
print(results)