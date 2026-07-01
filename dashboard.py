# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from backend import (
    load_resources,
    metadataRetrieval,
    mmrRetrieval,
    lexicalRetrieval,
    hybridRetrieval,
    structuredAnalysis,
    build_agent
)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veritasium Comments Analysis",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_resources():
    df, tfidf, tfidf_matrix, vectorstore = load_resources()
    agent = build_agent(vectorstore)
    return df, tfidf, tfidf_matrix, vectorstore, agent

with st.spinner("Loading analytics resources..."):
    df, tfidf, tfidf_matrix, vectorstore, agent = get_resources()

# ──────────────────────────────────────────────────────────────
# HEADER with LOGO
# ──────────────────────────────────────────────────────────────

log_col, title_col = st.columns([1, 5])

with log_col:
    # Uses the direct web link to the official high-res Veritasium logo
    st.image(
        "https://images.squarespace-cdn.com/content/v1/53ec3f51e4b0f5d3db27102b/1558639390029-T41BXMDTFBVDBAKFLSKK/veritasium+logo.png", 
        width=120
    )

with title_col:
    st.title("Veritasium YouTube Comments Analysis")
    st.header("Youtube video: The Bizarre Behavior of Rotating Bodies")
    st.header("Channel: Veritasium")
    st.markdown("Exploring viewer sentiment, topics, and discussions using NLP and RAG")
    st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW STATS
# ──────────────────────────────────────────────────────────────
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Comments", f"{len(df):,}")
col2.metric("Positive", f"{(df['tweetnlp_label']=='positive').sum():,}")
col3.metric("Negative", f"{(df['tweetnlp_label']=='negative').sum():,}")
col4.metric("Neutral", f"{(df['tweetnlp_label']=='neutral').sum():,}")

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 2 — SENTIMENT CHARTS
# ──────────────────────────────────────────────────────────────
st.header("Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = df['tweetnlp_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Sentiment Distribution (TweetNLP)',
        color='Sentiment',
        color_discrete_map={
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    vader_counts = df['vader_label'].value_counts().reset_index()
    vader_counts.columns = ['Sentiment', 'Count']
    fig2 = px.bar(
        vader_counts,
        x='Sentiment',
        y='Count',
        title='Sentiment Distribution (VADER)',
        color='Sentiment',
        color_discrete_map={
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 3 — TOPIC MODELING
# ──────────────────────────────────────────────────────────────
st.header("Topic Modeling")

col1, col2 = st.columns(2)

with col1:
    topic_counts = df['bertopic_topic'].value_counts().head(10).reset_index()
    topic_counts.columns = ['Topic', 'Count']
    topic_counts = topic_counts[topic_counts['Topic'] != -1]
    fig3 = px.bar(
        topic_counts,
        x='Topic',
        y='Count',
        title='Top 10 BERTopic Topics',
        labels={'Topic': 'Topic ID', 'Count': 'Number of Comments'}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    lda_counts = df['lda_topic'].value_counts().reset_index()
    lda_counts.columns = ['Topic', 'Count']
    fig4 = px.pie(
        lda_counts,
        values='Count',
        names='Topic',
        title='LDA Topic Distribution'
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 4 — COMMENT BROWSER (Metadata Retrieval)
# ──────────────────────────────────────────────────────────────
st.header("Browse Comments by Filter")

col1, col2, col3 = st.columns(3)

with col1:
    sentiment_filter = st.selectbox(
        "Sentiment",
        options=["All", "positive", "negative", "neutral"]
    )

with col2:
    min_likes = st.slider("Minimum Likes", min_value=0, max_value=100, value=0)

with col3:
    top_n = st.slider("Number of results", min_value=5, max_value=20, value=5)

if st.button("Filter Comments"):
    sentiment = None if sentiment_filter == "All" else sentiment_filter
    results = metadataRetrieval(
        df,
        sentiment=sentiment,
        min_likes=min_likes if min_likes > 0 else None,
        top_n=top_n
    )
    st.dataframe(results, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 5 — RETRIEVAL SEARCH
# ──────────────────────────────────────────────────────────────
st.header("Search Comments")

search_query = st.text_input("Enter a search query", placeholder="e.g. gravity force mass")

retrieval_type = st.radio(
    "Retrieval method",
    options=["Semantic (MMR)", "Lexical (TF-IDF)", "Hybrid"],
    horizontal=True
)

if st.button("Search") and search_query:
    with st.spinner("Searching records..."):
        if retrieval_type == "Semantic (MMR)":
            raw_docs = mmrRetrieval(vectorstore, search_query, k=5)
            # Standardize output appearance into an aesthetic tabular layout
            sem_df = pd.DataFrame(raw_docs, columns=['text_normalized'])
            st.dataframe(sem_df, use_container_width=True)

        elif retrieval_type == "Lexical (TF-IDF)":
            results = lexicalRetrieval(df, tfidf, tfidf_matrix, search_query, top_n=5)
            st.dataframe(results, use_container_width=True)

        elif retrieval_type == "Hybrid":
            results = hybridRetrieval(df, tfidf, tfidf_matrix, vectorstore, search_query, top_n=5)
            st.dataframe(results, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 6 — AI AGENT (QA + Summarization)
# ──────────────────────────────────────────────────────────────
st.header("Ask the AI Agent")
st.markdown("Ask a question or request a summary about the comments. The agent will automatically route your query.")

user_query = st.text_input(
    "Your question",
    placeholder="e.g. What do people think about gravity? / Summarize comments about the tennis racket effect"
)

if st.button("Ask Agent") and user_query:
    with st.spinner("Agent is thinking..."):
        result = agent.invoke({"query": user_query})
        st.markdown(f"**Intent detected:** `{result.get('intent', 'N/A')}`")
        st.markdown(f"**Key topic:** `{result.get('key_topic', 'N/A')}`")
        st.divider()
        st.markdown("### Answer")
        st.write(result.get('final_answer', 'No response generated.'))

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 7 — STRUCTURED ANALYSIS
# ──────────────────────────────────────────────────────────────
st.header("Structured Comment Analysis")
st.markdown("Paste a set of comments to get a structured breakdown of themes, sentiment, and concerns.")

comments_input = st.text_area(
    "Paste comments here (one per line)",
    height=150,
    placeholder="- gravity is just space being compressed\n- best physics video ever\n- the explanation was confusing"
)

if st.button("Analyze Comments") and comments_input:
    with st.spinner("Analyzing..."):
        result = structuredAnalysis(comments_input)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", result.main_sentiment)
            st.markdown("**Key Themes:**")
            for theme in result.key_themes:
                st.markdown(f"- {theme}")
        with col2:
            st.markdown("**Summary:**")
            st.write(result.summary)
            st.markdown("**Notable Concern:**")
            st.warning(result.notable_concern)