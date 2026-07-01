import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm  # This is the loading bar library

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Step 1: Loading text data columns directly from your Veritasium CSV...")
df = pd.read_csv("Processed comments/Veritasium_comments_modeling_topics_3.0.csv")
df['text'] = df['text'].fillna('')

print("Step 2: Initializing model configuration (mxbai-embed-large)...")
embeddings_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

BATCH_SIZE = 100
vectorstore = None
total_rows = len(df)

print("\nStep 3: Building Vector Store...")

# tqdm() wraps around our batch range to draw the live loading bar
for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing Comments", unit="batch"):
    batch_df = df.iloc[i : i + BATCH_SIZE]
    
    batch_texts = batch_df['text'].tolist()
    batch_metadatas = []
    
    for idx, row in batch_df.iterrows():
        batch_metadatas.append({
            "comment_id": int(row["comment_id"]),
            "author": str(row["author"]),
            "like_count": int(row["like_count"]),
            "vader_label": str(row["vader_label"]),
            "tweetnlp_label": str(row["tweetnlp_label"]),
            "lda_topic": int(row["lda_topic"]),
            "bertopic_topic": int(row["bertopic_topic"])
        })
    
    # Process incrementally
    if vectorstore is None:
        vectorstore = FAISS.from_texts(
            texts=batch_texts,
            embedding=embeddings_model,
            metadatas=batch_metadatas
        )
    else:
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas
        )

print("\nStep 4: Saving local FAISS vectorstore folder to disk...")
vectorstore.save_local("./veritasium_faiss_store_2")

print("\nSuccess! FAISS Index completely built and saved to './veritasium_faiss_store'!")