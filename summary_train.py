from flask import Flask, render_template
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def cluster_responses(responses, n_clusters=5):
    preprocessed_responses = [preprocess_text(resp) for resp in responses]
    vectorizer = CountVectorizer()
    response_vectors = vectorizer.fit_transform(preprocessed_responses)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(response_vectors)

    cluster_words = []
    feature_names = vectorizer.get_feature_names_out()
    for cluster_center in kmeans.cluster_centers_:
        top_words_idx = cluster_center.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        cluster_words.append(', '.join(top_words))

    return cluster_words


def generate_brief_summary(combined_feedback):
    all_words = ' '.join([preprocess_text(fb) for fb in combined_feedback if isinstance(fb, str)])
    word_freq = Counter(all_words.split())
    top_words = [word for word, _ in word_freq.most_common(10) if word not in ['good', 'bad']]

    positive_words = set(['good', 'best', 'excellent', 'great', 'impressive', 'supportive'])
    negative_words = set(['bad', 'worst', 'poor', 'terrible', 'needs', 'improvement'])

    positive_count = sum(1 for fb in combined_feedback if any(word in str(fb).lower() for word in positive_words))
    negative_count = sum(1 for fb in combined_feedback if any(word in str(fb).lower() for word in negative_words))
    total_responses = len([fb for fb in combined_feedback if str(fb).strip()])

    sentiment = "mostly positive" if positive_count > negative_count else "mostly negative" if negative_count > positive_count else "mixed"

    summary = f"The feedback is {sentiment}, with key themes including {', '.join(top_words[:3])}. "
    summary += f"Common points of discussion are {', '.join(top_words[3:6])}, "
    summary += f"while {', '.join(top_words[6:])} are also mentioned frequently."

    return summary


def summarize_feedback(df):
    combined_feedback = df['BestFeatures'].fillna('') + ' ' + df['ImprovementAreas'].fillna('')
    combined_feedback = combined_feedback.tolist()

    brief_summary = generate_brief_summary(combined_feedback)
    clusters = cluster_responses(combined_feedback, n_clusters=7)

    total_responses = len([fb for fb in combined_feedback if fb.strip()])
    response_rate = total_responses / len(df) * 100

    positive_words = set(['good', 'best', 'excellent', 'great', 'impressive', 'supportive'])
    negative_words = set(['bad', 'worst', 'poor', 'terrible', 'needs improvement'])

    positive_count = sum(1 for fb in combined_feedback if any(word in fb.lower() for word in positive_words))
    negative_count = sum(1 for fb in combined_feedback if any(word in fb.lower() for word in negative_words))

    return {
        'brief_summary': brief_summary,
        'clusters': clusters,
        'total_responses': total_responses,
        'response_rate': response_rate,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': total_responses - positive_count - negative_count
    }





