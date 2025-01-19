import pandas as pd
import re

# Function to clean tweet text
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string if the input is not a string
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert text to lowercase
    return text

def load_and_preprocess(filepath):
    # Read the CSV file with no header, as the file doesn't have a header row
    df = pd.read_csv(filepath, header=None)

    # Check the columns and the first few rows to confirm the structure
    print("Columns in the CSV:", df.columns)
    print("First few rows:", df.head())

    # Manually assign column names based on observed structure
    df.columns = ['id', 'game', 'sentiment', 'tweet_content']

    # Check if sentiment column exists
    if 'sentiment' not in df.columns:
        print("Sentiment column is missing!")
        return None

    # Clean the text
    df['cleaned_text'] = df['tweet_content'].apply(clean_text)

    # Map sentiment to numerical values
    sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    # Drop rows with missing sentiment values
    df = df.dropna(subset=['sentiment'])

    return df

if __name__ == "__main__":
    # Load and preprocess the data
    training_data = load_and_preprocess('../data/twitter_training.csv')
    
    if training_data is not None:
        # Save the cleaned data to a new CSV file
        training_data.to_csv('../data/cleaned_training_data.csv', index=False)
