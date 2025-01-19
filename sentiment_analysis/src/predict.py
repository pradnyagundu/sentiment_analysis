import pandas as pd
import joblib
import os

# Load the model and vectorizer
model_path = '/Users/pradnya/Desktop/sentiment_analysis/models/sentiment_model.pkl'
vectorizer_path = '/Users/pradnya/Desktop/sentiment_analysis/models/vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Please ensure they are in the 'models' folder.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load test data
test_data_path = '/Users/pradnya/Desktop/sentiment_analysis/data/twitter_test.csv'

if not os.path.exists(test_data_path):
    raise FileNotFoundError("Test data file not found. Please place 'twitter_test.csv' in the 'data' folder.")

data = pd.read_csv(test_data_path, header=0, names=['id', 'game', 'sentiment', 'tweet'])
print("First few rows of the test data:")
print(data.head())

# Predict
X_test = data['tweet']
X_test_vec = vectorizer.transform(X_test)
predictions = model.predict(X_test_vec)

# Display results
data['predicted_sentiment'] = predictions
print("Predictions:")
print(data[['tweet', 'predicted_sentiment']])
