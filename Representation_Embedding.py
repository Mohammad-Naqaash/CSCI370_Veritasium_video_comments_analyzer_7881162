import os
from langchain_ollama import OllamaEmbeddings

# Disable duplicate library flags (from Lab 4)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Initializing the high-performance model (mxbai-embed-large)...")
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")

# Quick test vector validation to make sure the local engine works
test_vector = embeddings_model.embed_query("Testing my project embedding engine.")
print(f"Success! Generated embedding dimensions: {len(test_vector)}")