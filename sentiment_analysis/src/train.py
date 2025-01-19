import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load the data
data = pd.read_csv('/Users/pradnya/Desktop/sentiment_analysis/data/twitter_training.csv', header=None, names=['id', 'game', 'sentiment', 'tweet'])

# Clean the data (drop rows with missing values in 'tweet' or 'sentiment')
data.dropna(subset=['tweet', 'sentiment'], inplace=True)

# Display the first few rows of the dataset to check
print("First few rows of the dataset:")
print(data.head())

# Split the data into features (X) and labels (y)
X = data['tweet']
y = data['sentiment']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline for TF-IDF vectorization and model training
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # Convert text data into numerical vectors
    ('classifier', MultinomialNB())     # Naive Bayes classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model's accuracy on the test set
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Save the trained model and the vectorizer to disk
joblib.dump(pipeline.named_steps['classifier'], '/Users/pradnya/Desktop/sentiment_analysis/models/sentiment_model.pkl')
joblib.dump(pipeline.named_steps['vectorizer'], '/Users/pradnya/Desktop/sentiment_analysis/models/vectorizer.pkl')

print("Model and vectorizer saved successfully.")
