import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from collections import Counter
import re
import os

# Load trained model and vectorizer with error handling
model_path = r"C:\Users\rahini r\Desktop\NLP\path\to\model\directory/best_sentimental_model.pkl"
vectorizer_path = r"C:\Users\rahini r\Desktop\NLP\path\to\model\directory/tf_idf_vectoriser.pkl"

if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit()

if not os.path.exists(vectorizer_path):
    print(f"Vectorizer file not found: {vectorizer_path}")
    exit()

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)  # Load TF-IDF vectorizer
except Exception as e:
    print(f"Error loading files: {str(e)}")
    exit()

# Load data for EDA
try:
    df = pd.read_csv(r"C:\Users\rahini r\Desktop\NLP\path\to\model\directory\processed_reviews.csv")
except FileNotFoundError:
    print("Processed reviews CSV file not found. Please check the path.")
    exit()

if "text" in df.columns and "sentiment" in df.columns:
    print("📌 Data Preview:")
    print(df.head())

    # Sentiment Distribution
    print("📊 Sentiment Distribution")
    sns.countplot(x=df["sentiment"], palette="viridis")
    plt.title("Sentiment Distribution")
    plt.show()

    # Sentiment by Rating
    print("📊 Sentiment by Rating")
    sentiment_by_rating = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_rating.plot(kind='barh', colormap='viridis')
    plt.ylabel("Rating")
    plt.xlabel("No of Reviews")
    plt.legend()
    plt.title("Sentiment by Rating")
    plt.show()

    # WordCloud for Sentiments
    print("📝 WordCloud for Sentiments")
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))
    sentiments = df["sentiment"].unique()

    for i, sentiment in enumerate(sentiments):
        reviews = df[df["sentiment"] == sentiment]["text"].astype('str')
        all_text = " ".join(reviews)
        cleaned_text = re.sub(r"[^\w\s]", "", all_text)
        cleaned_text = re.sub(r"\d+", "", cleaned_text)
        cleaned_text = cleaned_text.lower()
        words = cleaned_text.split()
        unwanted_keys = ["sentences", "tokenized_words", "stemmed_words", "lemmatized_words"]
        filtered_words = [word for word in words if word not in unwanted_keys and len(word) > 2]
        filtered_text = " ".join(filtered_words)
        wordcloud = WordCloud(width=500, height=300, background_color="white").generate(filtered_text)
        axes[i].imshow(wordcloud, interpolation="bilinear")
        axes[i].set_title(f"{sentiment} Sentiment")
        axes[i].axis("off")

    plt.show()

    # Sentiment Trend Over Time
    print("📅 Sentiment Trend Over Time")
    df["date"] = pd.to_datetime(df['date'], errors='coerce')
    sentiment_over_time = df.dropna(subset=['date']).groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
    sentiment_over_time.index = sentiment_over_time.index.to_timestamp()
    sentiment_over_time.plot(marker='o')
    plt.xlabel('Month')
    plt.ylabel('Sentiment Counts')
    plt.title("Sentiment Trend Over Time")
    plt.legend()
    plt.show()

    # Verified users vs reviews
    print("📝 Verified users vs reviews")
    verified_users_review = df.groupby(['verified_purchase', 'sentiment']).size().unstack(fill_value=0)
    verified_users_review.plot(kind='bar', colormap='viridis')
    plt.ylabel("Review counts")
    plt.title("Verified Users vs Reviews")
    plt.show()