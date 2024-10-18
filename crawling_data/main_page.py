import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st
import altair as alt
import numpy as np
import joblib
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as stopwords_scratch
import pickle

feature_bow = pickle.load(open("crawling_data/model/feature-bow.p",'rb'))
model_nb = pickle.load(open('crawling_data/model/model-nb.p', 'rb'))
model_nn = pickle.load(open('crawling_data/model/model-nn.p', 'rb'))

# Stopwords list (gabungan Indonesia & Inggris)
list_stopwords = stopwords_scratch.words('indonesian')
list_stopwords_en = stopwords_scratch.words('english')
list_stopwords.extend(list_stopwords_en)
list_stopwords.extend(['ya', 'yg', 'ga', 'yuk', 'dah', 'ngga', 'engga', 'ygy', 'tidak' ,'dan', 'ke'])
stopwords = list_stopwords

# Fungsi untuk cleansing teks
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'http\S+|www\S+|https\S+|t\.co\S+', '', string, flags=re.MULTILINE)
    string = re.sub(r'@\w+', '', string)
    string = re.sub(r'#', '', string)
    string = re.sub(r'\d+', '', string)
    string = re.sub(r'[^\w\s]', ' ', string)
    string = re.sub(r'\b(rt|t|co)\b', '', string)
    string = re.sub(r'\s+', ' ', string).strip()
    return string

# Preprocessing text
def preprocess_text(text):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    
    # Lakukan cleansing
    cleaned_text = cleansing(text)
    
    # Hilangkan stopword dari teks
    processed_text = ' '.join([word for word in cleaned_text.split() if word not in stopwords])
    
    return stopword_remover.remove(processed_text)

# Fungsi untuk membaca CSV yang di-upload dan membersihkannya
def read_upcsv(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'full_text' in data.columns:
            st.write("Data Preview:")
            st.dataframe(data.head())  # Show first few rows
            
            # Bersihkan data pada kolom 'full_text'
            data['processed_text'] = data['full_text'].apply(preprocess_text)
        else:
            st.error("Uploaded CSV must contain 'full_text'.")
            return None
    return data

def predict_sentiment(sent):
    # feature extraction
    text_feature = feature_bow.transform([sent])
    # predict
    return model_nb.predict(text_feature)[0]

# Generate word cloud menggunakan teks yang sudah diproses
def generate_wordcloud(data):
    text = ' '.join(data['processed_text'].astype(str).tolist())
    wc = WordCloud(background_color='black', max_words=500, width=800, height=400).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Main application
def main():
    st.title('Twitter Sentiment Analysis')

    menu = ["Sentiment Analysis", "Dataframe & Analysis", "Train Model"]
    choice = st.sidebar.selectbox("Menu", menu)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    data = read_upcsv(uploaded_file) if uploaded_file else None

    if data is not None:

        if choice == "Sentiment Analysis":
            st.subheader("Sentiment Analysis")

            # Preprocess and predict sentiment
            data['sentiment'] = data['processed_text'].apply(lambda x: predict_sentiment(x))

            # Display results
            st.write("### Sentiment Analysis Results")
            st.dataframe(data[['full_text', 'sentiment']])

            # Sentiment distribution
            sentiment_counts = data['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

        elif choice == "Dataframe & Analysis":
            st.subheader("Dataframe & Analysis")
            st.write("### Dataset")
            st.dataframe(data)

            # Generate Word Cloud
            st.write("### Word Cloud")
            generate_wordcloud(data)

        elif choice == "K-Means":
            st.subheader("Train Sentiment Analysis Model")
            data = predict_sentiment(data)
            st.success("Model trained successfully!")

if __name__ == '__main__':
    main()
