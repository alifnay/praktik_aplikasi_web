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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as stopwords_scratch
import pickle

# Load model dan features
feature_bow = pickle.load(open("crawling_data/model/feature-bow.p",'rb'))
model_nb = pickle.load(open('crawling_data/model/model-nb.p', 'rb'))
model_nn = pickle.load(open('crawling_data/model/model-nn.p', 'rb'))

# Stopword list (Indonesia & English)
list_stopwords = stopwords_scratch.words('indonesian')
list_stopwords_en = stopwords_scratch.words('english')
list_stopwords.extend(list_stopwords_en)
list_stopwords.extend(['ya', 'yg', 'ga', 'yuk', 'dah', 'ngga', 'engga', 'ygy', 'tidak' ,'dan', 'ke'])
stopwords = list_stopwords

# Cleansing function untuk text
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
    
    cleaned_text = cleansing(text)
    processed_text = ' '.join([word for word in cleaned_text.split() if word not in stopwords])
    
    return stopword_remover.remove(processed_text)

# Neutral threshold untuk predict sentiment
NEURAL_THRESHOLD = 0.5

# Function untuk predict sentiment
def predict_sentiment(sent):
    # Dapatkan fitur teks dari vectorizer
    text_feature = feature_bow.transform([sent])
    
    # Prediksi probabilitas dari model Naive Bayes
    probabilities = model_nb.predict_proba(text_feature)[0]
    
    # Dapatkan probabilitas tertinggi dan label prediksinya
    max_proba = np.max(probabilities)
    prediction = model_nb.classes_[np.argmax(probabilities)]
    
    # Tentukan apakah hasilnya 'neutral' berdasarkan threshold
    if max_proba < NEURAL_THRESHOLD:
        sentiment = 'neutral'
    else:
        sentiment = prediction
    
    return sentiment

# Generate word cloud
def generate_wordcloud(data):
    text = ' '.join(data['processed_text'].astype(str).tolist())
    wc = WordCloud(background_color='black', max_words=500, width=800, height=400).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# K-Means clustering
def kmeans_clustering(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['processed_text'])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X.toarray())
    data['pc1'] = principal_components[:, 0]
    data['pc2'] = principal_components[:, 1]

    return data, kmeans

# Visualize clusters
def visualize_clusters(data):
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x='pc1',
        y='pc2',
        color='cluster:N',
        tooltip=['full_text', 'cluster']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# Display cluster examples
def display_cluster_examples(data, n_examples=3):
    st.write("### Contoh dari tiap cluster")
    clusters = data['cluster'].unique()
    for cluster in clusters:
        st.write(f"#### Cluster {cluster}")
        examples = data[data['cluster'] == cluster].sample(n=min(n_examples, len(data[data['cluster'] == cluster]))).reset_index(drop=True)
        for _, row in examples.iterrows():
            st.write(f"- {row['full_text']}")

# Function untuk analisis input sentiment
def analyze_input_sentiment(text):
    processed_text = preprocess_text(text)
    sentiment = predict_sentiment(processed_text)
    
    if sentiment == 'positive':
        st.write(f"Sentiment: Positive 😄")
    elif sentiment == 'negative':
        st.write(f"Sentiment: Negative 😡")
    else:
        st.write(f"Sentiment: Neutral 😐")

# Main application
def main():
    st.title('Twitter Sentiment Analysis')

    menu = ["Sentiment Analysis", "Dataset dan WordCloud", "K-Means", "Input Sentiment Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    data = None
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'full_text' in data.columns:            
            data['processed_text'] = data['full_text'].apply(preprocess_text)
        else:
            st.error("Uploaded CSV must contain 'full_text'.")
            return

    if choice == "Sentiment Analysis" and data is not None:
        st.subheader("Sentiment Analysis")
        st.write("Data Preview:")
        st.dataframe(data.head())
        data['sentiment'] = data['processed_text'].apply(lambda x: predict_sentiment(x))
        st.write("### Sentiment Analysis Results")
        st.dataframe(data[['full_text', 'sentiment']])
        sentiment_counts = data['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    elif choice == "Dataset dan WordCloud" and data is not None:
        st.subheader("Dataset dan WordCloud")
        st.dataframe(data)
        st.write("### Word Cloud")
        generate_wordcloud(data)

    elif choice == "K-Means" and data is not None:
        st.subheader("K-Means Clustering")
        data, kmeans_model = kmeans_clustering(data)
        st.dataframe(data[['full_text', 'cluster']])
        st.write("### Cluster Visualization")
        visualize_clusters(data)
        display_cluster_examples(data)

    elif choice == "Input Sentiment Analysis":
        st.subheader("Analyze Sentiment of Your Own Text")
        user_input = st.text_area("Enter text to analyze sentiment:")
        if st.button("Analyze"):
            analyze_input_sentiment(user_input)

if __name__ == '__main__':
    main()
