import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import dspy
from typing import TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Veritasium video comments analyzer")
mlflow.langchain.autolog()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ──────────────────────────────────────────────────────────────
def load_resources():
    df = pd.read_csv("Processed comments/Veritasium_comments_modeling_topics_3.0.csv")
    df = df.dropna(subset=["text_normalized"]).reset_index(drop=True)

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['tokens_str'].fillna(''))

    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    vectorstore = FAISS.load_local(
        "./veritasium_faiss_store_2",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return df, tfidf, tfidf_matrix, vectorstore


# ──────────────────────────────────────────────────────────────
# RETRIEVAL FUNCTIONS
# ──────────────────────────────────────────────────────────────
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


def mmrRetrieval(vectorstore, question, k=5):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.3}
    )
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def lexicalRetrieval(df, tfidf, tfidf_matrix, query, top_n=5):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][['text_normalized', 'like_count']].copy()
    results['score'] = scores[top_indices]
    return results


def hybridRetrieval(df, tfidf, tfidf_matrix, vectorstore, query, top_n=5, pool_size=100, alpha=0.5):
    semantic_results = vectorstore.similarity_search_with_score(query, k=pool_size)
    semantic_scores = {doc.page_content: score for doc, score in semantic_results}
    candidate_texts = list(semantic_scores.keys())
    
    # Isolate vector matches inside parent frame
    candidate_rows = df[df['text_normalized'].isin(candidate_texts)].copy()
    query_vec = tfidf.transform([query])

    combined = []
    for idx, row in candidate_rows.iterrows():
        text = row['text_normalized']
        
        # FIX: Access the TF-IDF matrix using absolute dataframe row integer position index safely
        lex_score = cosine_similarity(query_vec, tfidf_matrix[idx]).flatten()[0]
        
        # Distance-to-similarity calibration
        sem_score = 1 / (1 + semantic_scores.get(text, 999))
        final_score = alpha * sem_score + (1 - alpha) * lex_score
        combined.append((text, row['like_count'], final_score))

    combined.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(combined[:top_n], columns=['text_normalized', 'like_count', 'hybrid_score'])


# ──────────────────────────────────────────────────────────────
# STRUCTURED PROMPTING
# ──────────────────────────────────────────────────────────────
class CommentAnalysis(BaseModel):
    main_sentiment: str
    key_themes: list[str]
    summary: str
    notable_concern: str

def structuredAnalysis(comments_text: str) -> CommentAnalysis:
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analyst reviewing YouTube comments about a physics video."),
        ("human", """
        Comments:
        {comments}

        Analyze these comments and return:
        - main_sentiment
        - key_themes (list of 3-5 themes)
        - summary
        - notable_concern
        """)
    ])

    chain = analysis_prompt | llm.with_structured_output(CommentAnalysis)
    return chain.invoke({"comments": comments_text})


# ──────────────────────────────────────────────────────────────
# AGENT ORCHESTRATION 
# ──────────────────────────────────────────────────────────────
def build_agent(vectorstore):
    lm = dspy.LM(
        'ollama/qwen2.5:7b',
        api_base='http://localhost:11434',
        api_key=None,
        temperature=0
    )

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

    class AgentState(TypedDict):
        query: str
        intent: str
        key_topic: str
        context: str
        final_answer: str

    def query_analysis_node(state: AgentState):
        with dspy.context(lm=lm):
            result = analyze(query=state["query"])
        return {"intent": result.intent, "key_topic": result.key_topic}

    def route_query(state: AgentState):
        return "summarization_agent" if "summar" in state["intent"].lower() else "qa_agent"

    def qa_agent(state: AgentState):
        docs = mmrRetrieval(vectorstore, state["query"], k=5)
        context = " ".join(docs)
        with dspy.context(lm=lm):
            result = gen_answer(question=state["query"], context=context)
        return {"context": context, "final_answer": result.answer}

    def summarization_agent(state: AgentState):
        docs = mmrRetrieval(vectorstore, state["key_topic"], k=10)
        comments_text = "\n".join(f"- {d}" for d in docs)
        with dspy.context(lm=lm):
            result = gen_summary(topic=state["key_topic"], comments=comments_text)
        return {"context": comments_text, "final_answer": result.summary}

    builder = StateGraph(AgentState)
    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("qa_agent", qa_agent)
    builder.add_node("summarization_agent", summarization_agent)
    builder.set_entry_point("query_analysis")
    builder.add_conditional_edges(
        "query_analysis", route_query,
        {"qa_agent": "qa_agent", "summarization_agent": "summarization_agent"}
    )
    builder.add_edge("qa_agent", END)
    builder.add_edge("summarization_agent", END)

    return builder.compile()