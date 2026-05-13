import streamlit as st
import pandas as pd
import re
import os
import kagglehub
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==========================================
# STAGE 1: SETUP, CACHING, & DATA LOADING
# ==========================================
st.set_page_config(page_title="RealTalk MVP", layout="wide")

@st.cache_resource
def load_models():
    """Load NLP models once and cache them to avoid reloading on every interaction."""
    # Stage 3: Sentiment Analysis Model
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Stage 4: Embeddings Model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return sentiment_analyzer, embedder

@st.cache_data
def load_and_preprocess_data():
    """Download Kaggle dataset and apply Stage 2 text preprocessing."""
    # Download dataset securely using kagglehub
    path = kagglehub.dataset_download("denizbilginn/google-maps-restaurant-reviews")
    df_raw = pd.read_csv(os.path.join(path, "reviews.csv"))
    
    # Select relevant columns
    df = df_raw[["business_name", "text", "rating"]].dropna().copy()
    df.columns = ["business", "review", "rating"]
    
    # Stage 2 Preprocessing logic
    def clean_text(text):
        text = text.lower()                                      # Lowercase
        text = re.sub(r"[^a-z0-9\s]", " ", text)                 # Clean punctuation/symbols
        text = re.sub(r"\s+", " ", text).strip()                 # Collapse whitespace
        return text
    
    df["clean"] = df["review"].astype(str).apply(clean_text)
    
    # Filter out reviews under 20 words to improve embedding quality
    df = df[df["clean"].apply(lambda x: len(x.split()) >= 20)].reset_index(drop=True)
    return df

def extract_keywords(docs, num_keywords=3):
    """Stage 6: Generate a simple label per cluster using keyword extraction."""
    if not docs:
        return "General"
    vectorizer = CountVectorizer(stop_words='english', max_features=num_keywords)
    try:
        vectorizer.fit(docs)
        return " & ".join(vectorizer.get_feature_names_out()).title()
    except Exception:
        return "Miscellaneous"

# ==========================================
# STAGE 7: DASHBOARD & FRONTEND
# ==========================================
st.title("🗣️ RealTalk MVP: Google Reviews Analyzer")
st.markdown("Unlock recurring themes and sentiment patterns from customer feedback in seconds.")

# Load backend stack
with st.spinner("Initializing Models and Loading Dataset..."):
    sentiment_analyzer, embedder = load_models()
    df = load_and_preprocess_data()

# Sidebar Configuration
st.sidebar.header("Model Parameters")
k_clusters = st.sidebar.slider("Number of Themes (K-Means)", min_value=2, max_value=5, value=3)

# Stage 1: Input (User Search)
business_list = sorted(df['business'].unique().tolist())
selected_business = st.selectbox("Search for a Business from the Dataset:", ["-- Select a Business --"] + business_list)

if selected_business != "-- Select a Business --":
    st.divider()
    
    # Filter dataset for selected business
    b_df = df[df['business'] == selected_business].copy()
    
    if len(b_df) < k_clusters:
        st.warning(f"Not enough reviews (≥20 words) for '{selected_business}' to run clustering. Please pick another business.")
    else:
        with st.spinner("Classifying sentiment, mapping embeddings, and clustering themes..."):
            
            # Stage 3: Sentiment Classification
            # We map distilBERT's output to standard Positive/Negative categories.
            sentiments = sentiment_analyzer(b_df['clean'].tolist(), truncation=True, max_length=512)
            b_df['sentiment'] = [s['label'].capitalize() for s in sentiments] 
            
            # Stage 4: Embedding
            embeddings = embedder.encode(b_df['clean'].tolist())
            
            # Stage 5: Theme Clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
            b_df['cluster'] = kmeans.fit_predict(embeddings)
            
            # Stage 6: Theme Labeling
            theme_labels = {}
            for cluster_id in range(k_clusters):
                cluster_docs = b_df[b_df['cluster'] == cluster_id]['clean'].tolist()
                theme_labels[cluster_id] = extract_keywords(cluster_docs, num_keywords=3)
                
            b_df['theme_name'] = b_df['cluster'].map(theme_labels)

        # ------------------------------------------
        # VISUALIZATIONS
        # ------------------------------------------
        st.subheader(f"📊 Dashboard: {selected_business} ({len(b_df)} robust reviews)")
        
        col1, col2 = st.columns(2)
        
        # Plotly: Sentiment Distribution Pie Chart
        with col1:
            sentiment_counts = b_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig_pie = px.pie(
                sentiment_counts, 
                names='Sentiment', 
                values='Count', 
                color='Sentiment',
                color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#1f77b4"},
                hole=0.4,
                title="Overall Sentiment Breakdown"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Matplotlib: Rating Distribution Bar Chart
        with col2:
            rating_counts = b_df['rating'].value_counts().sort_index()
            fig_bar, ax = plt.subplots(figsize=(6, 4.5))
            ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
            ax.set_xticks(range(1, 6))
            ax.set_xlabel("Star Rating")
            ax.set_ylabel("Number of Reviews")
            ax.set_title("Ratings Distribution")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_bar)

        # ------------------------------------------
        # THEME CARDS
        # ------------------------------------------
        st.subheader("💡 Identified Themes & Keyword Highlights")
        
        for cluster_id in range(k_clusters):
            t_name = theme_labels[cluster_id]
            theme_df = b_df[b_df['cluster'] == cluster_id]
            
            # Create a Streamlit Expander acting as a "Theme Card"
            with st.expander(f"📌 Theme: {t_name} — ({len(theme_df)} reviews)", expanded=True):
                
                # Show top 3 representative review snippets for this theme
                for _, row in theme_df.head(3).iterrows():
                    emoji = "🟢" if row['sentiment'] == "Positive" else "🔴"
                    st.markdown(f"**{emoji} {row['sentiment']}** (Rating: {row['rating']}/5) <br> *\"{row['review']}\"*", unsafe_allow_html=True)
                    st.markdown("---")
