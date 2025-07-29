import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # ✅ Load once

# Load dataset (first 2000 rows)
df = pd.read_csv(r"C:\Users\cheth\Downloads\archive (4)\IMDB Dataset.csv").head(2000)

# Clean function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text if word not in stop_words]  # ✅ use preloaded stopwords
    return ' '.join(text)

# Clean all reviews with progress
cleaned_reviews = []
for i, review in enumerate(df['review']):
    cleaned = clean_text(review)
    cleaned_reviews.append(cleaned)
    if i % 500 == 0:
        print(f"✅ Processed {i} reviews...")

df['clean_review'] = cleaned_reviews

# Convert sentiment to binary
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorize
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("🎉 Done! Model and vectorizer saved.")
