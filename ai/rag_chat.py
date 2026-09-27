import os

import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

# Start with 100 for Streamlit Cloud.
# Once everything works, you can increase this to 500.
NEW_REVIEWS = 100

TOP_K = 5

CACHE_FILE = "review_embeddings.parquet"

EMBEDDING_DIMENSION = 768


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not configured. "
        "Add it to Streamlit Cloud Secrets or your .env file."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# SNOWFLAKE
# ============================================================

def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    try:
        df = conn.cursor().execute(query).fetch_pandas_all()
    finally:
        conn.close()

    df.columns = [col.lower() for col in df.columns]

    # Remove null comments because embeddings need text
    df["comment"] = df["comment"].fillna("").astype(str)

    # Remove completely empty reviews
    df = df[df["comment"].str.strip() != ""].reset_index(drop=True)

    return df


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def embed(texts):
    """
    Generate embeddings for multiple texts in ONE API request.

    This is much better than making one API request per review.
    """

    if not texts:
        return []

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


# ============================================================
# LOAD REVIEWS + CREATE EMBEDDINGS
# ============================================================

@st.cache_data()
def load_reviews():

    # Use cached embeddings if available
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    # Otherwise get reviews from Snowflake
    df = read_reviews_from_snowflake()

    if df.empty:
        st.error("No reviews were returned from Snowflake.")
        st.stop()

    # Generate embeddings in a batch
    df["embedding"] = embed(
        df["comment"].tolist()
    )

    # Save locally so we don't regenerate embeddings
    # every time the Streamlit script reruns.
    df.to_parquet(CACHE_FILE)

    return df


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews, "
    f"answering with {CHAT_MODEL}"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    denominator = (
        np.linalg.norm(vec_a) *
        np.linalg.norm(vec_b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(vec_a, vec_b) / denominator


# ============================================================
# RETRIEVE SIMILAR REVIEWS
# ============================================================

def find_similar_reviews(question, df):

    # Convert the user's question into an embedding
    question_vector = embed([question])[0]

    scores = []

    # Compare question embedding against every review embedding
    for review_vector in df["embedding"]:
        score = cosine_similarity(
            question_vector,
            review_vector
        )

        scores.append(score)

    df = df.copy()

    df["score"] = scores

    # Return the most semantically similar reviews
    return df.nlargest(TOP_K, "score")


# ============================================================
# ASK GEMINI
# ============================================================

def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )

    system_prompt = """
You are answering questions about Zomato customer reviews.

Answer ONLY using the customer reviews provided below.

Do not invent information.

Be concise and directly answer the question.

If the provided reviews do not contain enough information
to answer the question, say that the available reviews
do not provide enough information.
"""

    user_prompt = f"""
Question:
{question}

Customer Reviews:
{context}
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"""
{system_prompt}

{user_prompt}
"""
    )

    return response.text


# ============================================================
# LOAD DATA
# ============================================================

review_df = load_reviews()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common complaints "
        "about delivery?"
    ),
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    with st.spinner("Searching reviews..."):

        # 1. Convert question → embedding
        # 2. Compare against review embeddings
        # 3. Retrieve top 5 reviews
        top_reviews = find_similar_reviews(
            question,
            review_df
        )

    with st.spinner("Generating answer..."):

        # 4. Send question + retrieved reviews to Gemini
        answer = ask_llm(
            question,
            top_reviews
        )

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown("### Answer")

    st.write(answer)

    # ========================================================
    # SOURCES
    # ========================================================

    with st.expander(
        "Reviews used to build this answer"
    ):

        st.dataframe(
            top_reviews[
                [
                    "city",
                    "rating",
                    "comment",
                    "score",
                ]
            ],
            hide_index=True,
        )
