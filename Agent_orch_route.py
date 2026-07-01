import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
import dspy
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

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

# ── LLM ────────────────────────────────────────────────────────────
lm = dspy.LM('ollama/qwen2.5:7b', api_base='http://localhost:11434', api_key=None, temperature=0)

# ── DSPy Signatures ─────────────────────────
class analyzeQuery(dspy.Signature):
    """analyze ONE user question about YouTube comments and classify its intent. Answer only for the single question given."""
    query = dspy.InputField(desc="a single user question or request about the comments dataset")
    intent = dspy.OutputField(desc="exactly one of: qa, summarization")
    key_topic = dspy.OutputField(desc="the main topic or keyword the user is asking about")

class getAnswer(dspy.Signature):
    """provide a short answer about the question given the context"""
    question = dspy.InputField(desc="question or a concept")
    context = dspy.InputField(desc="relevant context to the question")
    answer = dspy.OutputField(desc="short answer to question or better explanation")

class getSummary(dspy.Signature):
    """summarize the main themes and opinions expressed in a set of YouTube comments"""
    topic = dspy.InputField(desc="topic or theme to summarize comments about")
    comments = dspy.InputField(desc="a collection of YouTube comments related to the topic")
    summary = dspy.OutputField(desc="a concise summary of the main opinions, themes, and overall sentiment")

analyze = dspy.Predict(analyzeQuery)
gen_answer = dspy.Predict(getAnswer)
gen_summary = dspy.Predict(getSummary)

# ── LangGraph State ───────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    intent: str
    key_topic: str
    context: str
    final_answer: str

# ── Node 1: Query Analysis ───────────────────────────────────────
def query_analysis_node(state: AgentState):
    with dspy.context(lm=lm):
        result = analyze(query=state["query"])
    return {"intent": result.intent, "key_topic": result.key_topic}

# ── Router ────────────────────────────────────────────────────────
def route_query(state: AgentState):
    if "summar" in state["intent"]:
        return "summarization_agent"
    else:
        return "qa_agent"

# ── Node 2a: QA Agent ─────────────────────────────────────────────
def qa_agent(state: AgentState):
    docs = mmrRetrieval(vectorstore, state["query"], k=5)
    context = " ".join(docs)
    with dspy.context(lm=lm):
        result = gen_answer(question=state["query"], context=context)
    return {"context": context, "final_answer": result.answer}

# ── Node 2b: Summarization Agent ─────────────────────────────────
def summarization_agent(state: AgentState):
    docs = mmrRetrieval(vectorstore, state["key_topic"], k=10)
    comments_text = "\n".join(f"- {d}" for d in docs)
    with dspy.context(lm=lm):
        result = gen_summary(topic=state["key_topic"], comments=comments_text)
    return {"context": comments_text, "final_answer": result.summary}

# ── Build Graph ───────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("query_analysis", query_analysis_node)
builder.add_node("qa_agent", qa_agent)
builder.add_node("summarization_agent", summarization_agent)

builder.set_entry_point("query_analysis")
builder.add_conditional_edges(
    "query_analysis",
    route_query,
    {"qa_agent": "qa_agent", "summarization_agent": "summarization_agent"}
)
builder.add_edge("qa_agent", END)
builder.add_edge("summarization_agent", END)

graph = builder.compile()

# ── Test it ──────────────────────────────────────────────────────
test_queries = [
    "What do people think about gravity?",
    "Summarize the comments about the tennis racket effect"
]

for q in test_queries:
    result = graph.invoke({"query": q})
    print(f"Query: {q}")
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['final_answer']}\n")