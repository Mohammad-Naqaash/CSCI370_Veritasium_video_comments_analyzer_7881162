import dspy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

lm = dspy.LM(
    'ollama/llama3.2:3b',
    api_base='http://localhost:11434',
    api_key=None,
    temperature=0
)

class analyzeQuery(dspy.Signature):
    """analyze ONE user question about YouTube comments and classify its intent. Answer only for the single question given."""
    query = dspy.InputField(desc="a single user question or request about the comments dataset")
    intent = dspy.OutputField(desc="exactly one of: qa, summarization, sentiment_lookup, topic_lookup")
    key_topic = dspy.OutputField(desc="the main topic or keyword the user is asking about")

analyze = dspy.Predict(analyzeQuery)

test_queries = [
    "What do people think about gravity?",
    "Summarize the comments about the tennis racket effect",
    "How many comments are negative about the video?",
    "What topics are people discussing the most?"
]

with dspy.context(lm=lm) as ctx:
    for q in test_queries:
        try:
            result = analyze(query=q)
            print(f"Query: {q}")
            print(f"  → Intent: {result.intent}")
            print(f"  → Key topic: {result.key_topic}\n")
        except Exception as e:
            print(f"Query: {q}")
            print(f"  → Failed to parse: {e}\n")