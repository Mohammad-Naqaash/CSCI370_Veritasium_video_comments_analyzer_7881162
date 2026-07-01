import os
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# ── Define structured output schema ──
class CommentAnalysis(BaseModel):
    main_sentiment: str        # overall sentiment: positive/negative/neutral
    key_themes: list[str]      # list of main topics/themes discussed
    summary: str                # short summary
    notable_concern: str        # any notable complaint or confusion point

# ── Build prompt + structured chain ───────────────────────────────
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

# ── Test it ──────────────────────────────────────────────────────
sample_comments = """
- gravity is just space being compressed, not a real force
- this blew my mind, never understood it like this before
- the explanation skipped over some math, kind of confusing
- I think the simulation was wrong about the spin axis
- best physics video I've seen all year
"""

result = chain.invoke({"comments": sample_comments})

print("Main Sentiment:", result.main_sentiment)
print("Key Themes:", result.key_themes)
print("Summary:", result.summary)
print("Notable Concern:", result.notable_concern)