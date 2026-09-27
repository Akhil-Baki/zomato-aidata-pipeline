import os

import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

NEW_REVIEWS = 100
TOP_K = 5

EMBEDDING_DIMENSION = 768

CACHE_FILE = "review_embeddings.parquet"


# ============================================================
# SECRETS
# ============================================================

def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing.")
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def get_snowflake_connection():

    return snowflake.connector.connect(
        account=get_secret("SNOWFLAKE_ACCOUNT"),
        user=get_secret("SNOWFLAKE_USER"),
        password=get_secret("SNOWFLAKE_PASSWORD"),
        warehouse=get_secret("SNOWFLAKE_WAREHOUSE"),
        database=get_secret("SNOWFLAKE_DATABASE"),
        schema=get_secret("SNOWFLAKE_SCHEMA"),
    )


# ============================================================
# READ REVIEWS FROM SNOWFLAKE
# ============================================================

def read_reviews_from_snowflake():

    conn = get_snowflake_connection()

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
        cursor = conn.cursor()

        try:
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
        finally:
            cursor.close()

    finally:
        conn.close()

    df.columns = [
        col.lower()
        for col in df.columns
    ]

    df["comment"] = (
        df["comment"]
        .fillna("")
        .astype(str)
    )

    df = df[
        df["comment"].str.strip() != ""
    ].reset_index(drop=True)

    return df


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def embed(texts):

    if not texts:
        return []

    try:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
        )

        embeddings = [
            embedding.values
            for embedding in response.embeddings
        ]

        return embeddings

    except Exception as e:

        st.error("Gemini embedding request failed.")

        st.write("Error type:", type(e).__name__)

        st.code(str(e))

        st.stop()


# ============================================================
# LOAD REVIEWS AND EMBEDDINGS
# ============================================================

@st.cache_data
def load_reviews():

    if os.path.exists(CACHE_FILE):

        try:

            df = pd.read_parquet(CACHE_FILE)

            # Make sure cached vectors use the current dimensions
            if (
                not df.empty
                and "embedding" in df.columns
                and len(df["embedding"].iloc[0])
                == EMBEDDING_DIMENSION
            ):
                return df

        except Exception:
            pass

    df = read_reviews_from_snowflake()

    if df.empty:
        st.error("No reviews found in Snowflake.")
        st.stop()

    with st.spinner("Generating review embeddings..."):

        embeddings = embed(
            df["comment"].tolist()
        )

    if len(embeddings) != len(df):

        st.error(
            "The number of embeddings does not "
            "match the number of reviews."
        )

        st.stop()

    df["embedding"] = embeddings

    df.to_parquet(CACHE_FILE)

    return df


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):

    vec_a = np.asarray(
        vec_a,
        dtype=float
    )

    vec_b = np.asarray(
        vec_b,
        dtype=float
    )

    denominator = (
        np.linalg.norm(vec_a)
        * np.linalg.norm(vec_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vec_a, vec_b) / denominator
    )


# ============================================================
# RETRIEVE RELEVANT REVIEWS
# ============================================================

def find_similar_reviews(question, df):

    # Convert question into an embedding
    question_vector = embed([question])[0]

    scores = []

    # Compare question with every stored review embedding
    for review_vector in df["embedding"]:

        score = cosine_similarity(
            question_vector,
            review_vector
        )

        scores.append(score)

    result = df.copy()

    result["score"] = scores

    # Retrieve top K relevant reviews
    return result.nlargest(
        TOP_K,
        "score"
    )


# ============================================================
# GENERATE ANSWER USING GEMINI
# ============================================================

def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )

    prompt = f"""
You are a Zomato customer review analytics assistant.

Your job is to answer questions based ONLY on
the customer reviews provided.

Rules:

1. Use only the reviews provided.
2. Do not invent facts.
3. Keep the answer clear and concise.
4. Identify patterns in customer feedback where possible.
5. If the reviews do not contain enough information,
   clearly say so.
6. Do not claim that the retrieved reviews represent
   all Zomato customers.

USER QUESTION:
{question}

CUSTOMER REVIEWS:
{context}

ANSWER:
"""

    try:

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )

        if not response.text:

            return (
                "Gemini returned an empty response. "
                "Please try again."
            )

        return response.text

    except Exception as e:

        st.error("Gemini answer generation failed.")

        st.write(
            "**Error type:**",
            type(e).__name__
        )

        # Show the HTTP status if available
        status_code = getattr(
            e,
            "code",
            None
        )

        if status_code is not None:

            st.write(
                "**HTTP status:**",
                status_code
            )

        # Display the actual Gemini API error
        st.write("**Gemini API error:**")

        st.code(str(e))

        st.info(
            "If this is a 500 or 503 error, "
            "it may be a temporary Gemini server issue. "
            "If it is a 429 error, check your API quota."
        )

        st.stop()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Zomato Review RAG",
    page_icon="🍽️",
    layout="wide"
)

st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews, "
    f"answering with {CHAT_MODEL}"
)


# ============================================================
# INITIALIZE REVIEWS
# ============================================================

review_df = load_reviews()

st.success(
    f"Loaded {len(review_df)} customer reviews."
)


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "What are customers complaining "
        "about the most?"
    ),
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    # STEP 1: Retrieve relevant reviews
    with st.spinner("Finding relevant reviews..."):

        top_reviews = find_similar_reviews(
            question,
            review_df
        )

    # STEP 2: Generate answer
    with st.spinner("Generating AI answer..."):

        answer = ask_llm(
            question,
            top_reviews
        )

    # STEP 3: Display answer
    st.markdown("### AI Answer")

    st.write(answer)

    # STEP 4: Display retrieved reviews
    with st.expander(
        "Reviews used to generate this answer"
    ):

        st.dataframe(
            top_reviews[
                [
                    "city",
                    "rating",
                    "comment",
                    "score"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )
