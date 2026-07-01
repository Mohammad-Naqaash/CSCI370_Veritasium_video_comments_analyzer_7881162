import pandas as pd

df = pd.read_csv("Processed comments/Veritasium_comments_modeling_topics_3.0.csv")
df = df.dropna(subset=["text_normalized"])

def metadataRetrieval(df, sentiment=None, lda_topic=None, bertopic_topic=None, min_likes=None, top_n=5):
    result = df.copy()

    if sentiment is not None:
        result = result[result["tweetnlp_label"] == sentiment]
    if lda_topic is not None:
        result = result[result["lda_topic"] == lda_topic]
    if bertopic_topic is not None:
        result = result[result["bertopic_topic"] == bertopic_topic]
    if min_likes is not None:
        result = result[result["like_count"] >= min_likes]

    result = result.sort_values(by="like_count", ascending=False)
    return result[["text_normalized", "like_count", "tweetnlp_label", "lda_topic", "bertopic_topic"]].head(top_n)

results = metadataRetrieval(df, sentiment="negative", min_likes=10, top_n=5)
print(results)