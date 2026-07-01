import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load comments ─────────────────────────────────────────────
df = pd.read_csv("Processed comments/Veritasium_comments_modeling_topics_3.0.csv")
df = df.dropna(subset=["text_normalized"])

# ── Build TF-IDF matrix over all comments ───────────────────────
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['tokens_str'].fillna(''))

# ── Lexical retrieval function ───────────────────────────────────
def lexicalRetrieval(query, top_n=5):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices][['text_normalized', 'like_count']].copy()
    results['score'] = scores[top_indices]
    return results

# ── Test it ──────────────────────────────────────────────────────
query = "mass gravity force"
results = lexicalRetrieval(query, top_n=5)
print(results)