import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
from flask import Flask, render_template, jsonify
from io import BytesIO
import base64
from test import load_data,load_assignment_records,calculate_scores,analyze_courses

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Specify the fixed CSV file path
CSV_FILE_PATH = 'feedback_data.csv'  # Make sure this file exists in the same directory as your Flask app


def categorize_sentiment(text):
    if pd.isna(text):
        return 'Neutral'
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Good'
    elif score < -0.05:
        return 'Bad'
    else:
        return 'Neutral'


def combine_sentiments(row):
    if 'Bad' in [row['BestFeatures_Sentiment'], row['ImprovementAreas_Sentiment']]:
        return 'Bad'
    elif 'Good' in [row['BestFeatures_Sentiment'], row['ImprovementAreas_Sentiment']]:
        return 'Good'
    else:
        return 'Neutral'


def get_top_words(texts, sentiment, n=10):
    words = ' '.join(texts).lower().split()
    return Counter(words).most_common(n)


def analyze_feedback():
    df = pd.read_csv(CSV_FILE_PATH, usecols=['BestFeatures', 'ImprovementAreas'])

    df['BestFeatures_Sentiment'] = df['BestFeatures'].apply(categorize_sentiment)
    df['ImprovementAreas_Sentiment'] = df['ImprovementAreas'].apply(categorize_sentiment)
    df['Combined_Sentiment'] = df.apply(combine_sentiments, axis=1)

    results = {}

    for sentiment in ['Good', 'Bad', 'Neutral']:
        best_features = df[df['Combined_Sentiment'] == sentiment]['BestFeatures']
        improvement_areas = df[df['Combined_Sentiment'] == sentiment]['ImprovementAreas']
        top_words = get_top_words(pd.concat([best_features, improvement_areas]), sentiment)
        examples = df[df['Combined_Sentiment'] == sentiment].head(3)

        results[sentiment] = {
            'top_words': top_words,
            'examples': examples[['BestFeatures', 'ImprovementAreas']].to_dict('records')
        }

    sentiment_distribution = df['Combined_Sentiment'].value_counts().to_dict()

    plt.figure(figsize=(10, 6))
    df['Combined_Sentiment'].value_counts().plot(kind='bar')
    plt.title('Distribution of Feedback Sentiments')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Feedbacks')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return results, sentiment_distribution, plot_url

@app.route('/')
def index():
    results, distribution, plot_url = analyze_feedback()
    return render_template('clusters.html', results=results, distribution=distribution, plot_url=plot_url)

@app.route('/comments_feedback')
def comments_feedback():
    feedback_data = load_data('feedback_data.csv')
    assignment_records = load_assignment_records('assignment_records.csv')
    scores = calculate_scores(feedback_data)
    results = analyze_courses(scores, assignment_records)
    return render_template('comments_feedback.html', **results)

@app.route('/api/analyze')
def api_analyze():
    feedback_data = load_data('feedback_data.csv')
    assignment_records = load_assignment_records('assignment_records.csv')
    scores = calculate_scores(feedback_data)
    results = analyze_courses(scores, assignment_records)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)