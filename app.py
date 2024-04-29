import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib 


# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|http?://\S+', '', text)
    # Remove special characters, numbers, and punctuations
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word not in stop_words])
    return text


# Load the pre-trained model and vectorizer
model_path = 'logistic_speech_model.pkl'  # Change this path to your model file path
vectorizer_path = 'tfidf_speech.pkl'  # Change this path to your vectorizer file path
lr_model = LogisticRegression()
tfidf_vectorizer = TfidfVectorizer()


# Load the saved model and vectorizer
lr_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)


# Streamlit app
def main():
    st.title("Hate Speech Detection 🆎")

    # Input text from user
    user_input = st.text_area("Enter a speech:", "")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            user_input_processed = preprocess_text(user_input)
            # Vectorize the processed text
            user_input_vectorized = tfidf_vectorizer.transform([user_input_processed])
            # Make prediction
            prediction = lr_model.predict(user_input_vectorized)

            
            if prediction[0] == 1:
                st.subheader("This is a hate speech.")
            else:
                st.subheader("This is a normal speech.")
        else:
            st.warning("Please enter a text.")



if __name__ == "__main__":
    main()
