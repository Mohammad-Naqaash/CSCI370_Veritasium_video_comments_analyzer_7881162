import dspy
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Load FAISS vectorstore ───────────────────────────────────────
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
vectorstore = FAISS.load_local("./veritasium_faiss_store_2", embeddings, allow_dangerous_deserialization=True)

# ── MMR retrieval function ────────────────────────────
def mmrRetrieval(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.3}
    )
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]

# ── Set up local LLM ──────────────────────────────────
lm = dspy.LM(
    'ollama/llama3.2:3b',
    api_base='http://localhost:11434',
    api_key=None,
    temperature=0.9
)

# ── Define the answer signature  ───────────────────────
class getAnswer(dspy.Signature):
    """provide a short answer about the question given the context"""
    question = dspy.InputField(desc="question or a concept")
    context = dspy.InputField(desc="relevant context to the question")
    answer = dspy.OutputField(desc="short answer to question or better explanation")

gen_answer = dspy.Predict(getAnswer)

# ── Run a test question ──────────────────────────────────────────
question = "What do people think about gravity in the comments?"
docs = mmrRetrieval(vectorstore, question)
context = " ".join(docs)

with dspy.context(lm=lm) as ctx:
    result = gen_answer(question=question, context=context)
    print("Question:", question)
    print("\nAnswer:", result.answer)