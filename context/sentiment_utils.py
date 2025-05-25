# context/sentiment_utils.py

import PyPDF2
import pandas as pd
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

afinn = Afinn()
vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join([page.extract_text() for page in reader.pages])

def analyze_text(text):
    word_count = len(text.split())
    afinn_score = afinn.score(text)
    afinn_adjusted = (afinn_score / word_count) * 100 if word_count else 0
    vader_scores = vader.polarity_scores(text)

    return {
        "word_count": word_count,
        "afinn_score": afinn_score,
        "afinn_adjusted": afinn_adjusted,
        **vader_scores
    }

def run_sentiment_analysis(pdf_path):
    text = extract_text(pdf_path)
    result = analyze_text(text)
    return pd.DataFrame([result])
