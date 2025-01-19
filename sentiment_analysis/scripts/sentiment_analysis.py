import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# File paths
DATA_PATH = '/Users/pradnya/Desktop/sentiment_analysis/data'
MODEL_PATH = '../models/sentiment_model.pkl'
VECTORIZER_PATH = '../models/vectorizer.pkl'

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Dataset not found at {DATA_PATH}. Please check the file path.")
    exit()

# Check dataset structure
print("Dataset Columns:", df.columns)

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", '', text)  # Remove punctuations and numbers
    text = text.lower()  # Lowercase
    return text

# Check and clean the dataset
if 'Tweet content' in df.columns:
    df['cleaned_text'] = df['Tweet content'].apply(clean_text)
elif 'text' in df.columns:  # Example alternative column name
    df['cleaned_text'] = df['text'].apply(clean_text)
else:
    print("No column containing tweet content found.")
    exit()

# Define features and target
X = df['cleaned_text']
y = df['sentiment']  # Replace 'sentiment' with the actual column name for labels

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
os.makedirs('../models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model training completed and saved!")

# Evaluate the model
y_pred = model.predict(X_val_vec)
print(classification_report(y_val, y_pred))

# Example prediction
sample_text = "I love this product!"
sample_text_vec = vectorizer.transform([sample_text])
prediction = model.predict(sample_text_vec)
print(f"Predicted sentiment for '{sample_text}': {prediction[0]}")
