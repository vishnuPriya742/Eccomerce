from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("./dataset/products.csv")

# Preprocess function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Create combined text column
df["text"] = (df["name"].astype(str) + " " + df["description"].astype(str)).apply(preprocess)

# Fit TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(df["text"])

# Recommendation function
def search_and_recommend(query, top_n=5):  # default changed to 5
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])

    all_vectors = vectorizer.transform(df["text"])
    cos_sim = cosine_similarity(query_vec, all_vectors).flatten()
    df["similarity"] = cos_sim

    sorted_products = df.sort_values(
        by=["similarity", "rating"], ascending=[False, False]
    )

    results = sorted_products.head(top_n)[
        ["id", "name", "category", "price", "rating", "stock"]
    ].to_dict(orient="records")

    top_category = results[0]["category"] if results else None
    return top_category, results

# Routes
@app.route("/", methods=["GET"])
def home():
    return "Welcome to E-commerce Recommender! Use POST /recommend with JSON { 'query': 'product name' }"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400
    try:
        top_category, results = search_and_recommend(query, top_n=5)  # fetch top 5
        return jsonify({
            "query": query,
            "top_category": top_category,
            "recommendations": results
        })
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

