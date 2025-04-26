import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Load and prepare the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        # Keep only relevant columns and rename them
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Text preprocessing with error handling
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Exploratory Data Analysis
def perform_eda(df):
    try:
        # Class distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='label', data=df)
        plt.title('Distribution of Spam vs Ham Messages')
        plt.show()
        
        # Message length analysis
        df['message_length'] = df['message'].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
        plt.title('Message Length Distribution by Label')
        plt.show()
        
        # Word clouds
        spam_words = ' '.join(df[df['label'] == 'spam']['message'])
        ham_words = ' '.join(df[df['label'] == 'ham']['message'])
        
        plt.figure(figsize=(12, 6))
        wordcloud = WordCloud(width=600, height=300).generate(spam_words)
        plt.imshow(wordcloud)
        plt.title('Spam Messages Word Cloud')
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        wordcloud = WordCloud(width=600, height=300).generate(ham_words)
        plt.imshow(wordcloud)
        plt.title('Ham Messages Word Cloud')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error during EDA: {e}")

# Train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    try:
        # Initialize and train the model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        
        return model
    except Exception as e:
        print(f"Error during model training/evaluation: {e}")
        return None

# Main function
def main():
    # Download required NLTK resources
    download_nltk_resources()
    
    # Load data
    file_path = 'spam.csv'
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Preprocess the messages
    print("Preprocessing messages...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Perform EDA
    print("Performing exploratory data analysis...")
    perform_eda(df)
    
    # Split data into train and test sets
    X = df['processed_message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize the text data using TF-IDF
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train and evaluate model
    print("Training model...")
    model = train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test)
    
    if model:
        print("\nModel trained successfully!")
    else:
        print("\nModel training failed.")

if __name__ == "__main__":
    main()