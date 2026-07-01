import dspy
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Load FAISS vectorstore ───────────────────────────────────────
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
vectorstore = FAISS.load_local("./veritasium_faiss_store_2", embeddings, allow_dangerous_deserialization=True)

def mmrRetrieval(vectorstore, question, k=10):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.3}
    )
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]

# ── Set up local LLM ──────────────────────────────────────────────
lm = dspy.LM(
    'ollama/llama3.2:3b',
    api_base='http://localhost:11434',
    api_key=None,
    temperature=0.7
)

# ── Define summarization signature ────────────────────────────────
class getSummary(dspy.Signature):
    """summarize the main themes and opinions expressed in a set of YouTube comments"""
    topic = dspy.InputField(desc="topic or theme to summarize comments about")
    comments = dspy.InputField(desc="a collection of YouTube comments related to the topic")
    summary = dspy.OutputField(desc="a concise summary of the main opinions, themes, and overall sentiment")

gen_summary = dspy.Predict(getSummary)

# ── Run a test summarization ──────────────────────────────────────
topic = "the tennis racket effect / Dzhanibekov effect"
docs = mmrRetrieval(vectorstore, topic, k=10)
comments_text = "\n".join(f"- {d}" for d in docs)

with dspy.context(lm=lm) as ctx:
    result = gen_summary(topic=topic, comments=comments_text)
    print("Topic:", topic)
    print("\nSummary:", result.summary)