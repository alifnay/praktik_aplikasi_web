import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st
import altair as alt
import numpy as np
import joblib
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

# Melakukan preprocessing text
def preprocess_text(text):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    return stopword_remover.remove(text.lower())

# NEURAL THRESHOLD untuk mendeteksi kata netral
NEURAL_THRESHOLD = 0.05

# Load model
@st.cache_resource
def load_model():
    return joblib.load('crawling_data\model\sentiment_mode.joblib')

# Fungsi untuk wordcloud
def generate_wordcloud(data):
    # Combine the full_text column into one large string
    text = ' '.join(data['full_text'].astype(str).tolist())

    # Generate word cloud
    wc = WordCloud(background_color='black', max_words=500, width=800, height=400).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Fungsi untuk K-Means clustering
def perform_kmeans(df):
    return df

# Main page
def main():
    
    menu = ["Sentiment Analysys", "Dataframe & Analysis", "KMeans Clustering"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load model dari file
    model = load_model()
    
    if choice == "Sentiment Analysys":
        st.subheader("Sentiment Analysys")
        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')
        
        # layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                
                # Preprocess input text
                processed_text = preprocess_text(raw_text)
                
                # Analyze sentiment menggunakan model pipeline
                prediction = model.predict([processed_text])  # Membuat prediction
                sentiment_prob = model.predict_proba([processed_text])  # Dapatkan probabilitas
                max_proba = np.max(sentiment_prob)  # Ambil probabilitas tertinggi
                
                # Menentukan sentimen berdasarkan probabilitas
                if max_proba < NEURAL_THRESHOLD:
                    sentiment = 'neutral'
                else:
                    sentiment = prediction[0]  # Mengambil prediksi sentiment
                    
                # Menampilkan input text
                st.write(f"Input Text: {processed_text}")
                
                # Menampilkan hasil
                if sentiment == 'positive':
                    st.markdown("Sentiment: Positive :smiley: ")
                elif sentiment == 'negative':
                    st.markdown("Sentiment: Negative :angry: ")
                else:
                    st.markdown("Sentiment: Neutral :neutral_face: ")
                
                # Membuat dataframe untuk hasil 
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualisasi
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            # Token sentiment
            with col2:
                st.info("Token Sentiment")
                token_sentiment = analyze_sentiment(processed_text, model)
                st.write(token_sentiment)


    elif choice == "Dataframe & Analysis":
        st.subheader("Dataframe & Analysis")
        
        data = pd.read_csv('crawling_data\data\data_crawling_label.csv')

        # Display dataset
        st.write("### Dataset")
        st.dataframe(data)

        # Menghitung distribusi sentiment
        sentiment_counts = data['sentiment'].value_counts()
        st.write("### Distribusi Sentimen")
        st.bar_chart(sentiment_counts)

        # Display number of positive, negative, and neutral sentiments
        st.write("#### Jumlah Sentiment")
        st.write(sentiment_counts)

        # Generate Word Cloud
        st.write("### Word Cloud")
        generate_wordcloud(data)
    
    elif choice == "KMeans Clustering":
        st.subheader("KMeans Clustering")
        
        data = pd.read_csv('crawling_data\data\data_crawling_label.csv')

        # Convert text data to list
        text_data = data['full_text'].to_list()

        # Vectorization
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text_data)

        # KMeans clustering
        true_k = 8
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

        # Display clusters
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        cluster_info = ""
        for i in range(true_k):
            cluster_info += f"Cluster {i}: " + ', '.join([terms[ind] for ind in order_centroids[i, :10]]) + "\n"
        st.write("#### Cluster: ")
        st.text(cluster_info)

        score = silhouette_score(X, model.labels_)
        st.write(f"#### Silhouette Score: {score:.2f}")

        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(X.toarray())
        reduced_cluster_centers = pca.transform(model.cluster_centers_)

        # Plotting the clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
        plt.title('KMeans Clustering Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter)
        plt.legend()
        st.pyplot(plt)

# Function untuk mengconvert sentiment ke dalam dataframe
def convert_to_df(sentiment):
    sentiment_dict = {'Sentiment': sentiment}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Function untuk menganalyze sentiment (mengelompokkan jenis kata)
def analyze_sentiment(docx, model):
    # Tokenisasi teks menjadi kata-kata
    tokens = docx.split()
    
    # Membuat list untuk menyimpan kata berdasarkan sentimen
    positive_words = []
    negative_words = []
    neutral_words = []
    
    for token in tokens:
        cleaned_token = token.lower().strip('.,!?"\'')  # Menghapus tanda baca dan mengubah menjadi huruf kecil
        
        # Cek apakah token ada dalam DataFrame
        sentiment_prob = model.predict_proba([cleaned_token])  # Dapatkan probabilitas
        max_proba = np.max(sentiment_prob)  # Ambil probabilitas tertinggi
        
        if max_proba < NEURAL_THRESHOLD:
            neutral_words.append(cleaned_token)  # Anggap netral jika di bawah ambang batas
        else:
            sentiment = model.classes_[np.argmax(sentiment_prob)]
            if sentiment == 'positive':
                positive_words.append(cleaned_token)
            elif sentiment == 'negative':
                negative_words.append(cleaned_token)

    # Mengembalikan kata-kata yang dikelompokkan berdasarkan sentimen
    result = {
        'positive': positive_words,
        'negative': negative_words,
        'neutral': neutral_words
    }
    return result

if __name__ == '__main__':
    main()
