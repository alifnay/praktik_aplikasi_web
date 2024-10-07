import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st
import altair as alt
import numpy as np

# Mengambil dataset dan preprocess
@st.cache_data
def load_data():
    # Mengambil dataset yang akan digunakan
    df1 = pd.read_csv('minggu_7\\dataset_komentar_instagram_cyberbullying.csv')
    df2 = pd.read_csv('minggu_7\\dataset_tweet_sentimen_tayangan_tv.csv')
    df3 = pd.read_csv('minggu_7\\dataset_tweet_sentiment_cellular_service_provider.csv')
    df4 = pd.read_csv('minggu_7\\dataset_tweet_sentiment_opini_film.csv')
    df5 = pd.read_csv('minggu_7\\master_emoji.csv')
    # Memilih dan mengubah nama kolom yang relevan
    df1 = df1[['Id', 'Sentiment', 'Instagram Comment Text']].rename(columns={'Instagram Comment Text': 'text'})
    df2 = df2[['Id', 'Sentiment', 'Text Tweet']].rename(columns={'Text Tweet': 'text'})
    df3 = df3[['Id', 'Sentiment', 'Text Tweet']].rename(columns={'Text Tweet': 'text'})
    df4 = df4[['Id', 'Sentiment', 'Text Tweet']].rename(columns={'Text Tweet': 'text'})
    df5 = df5[['ID', 'Sentiment', 'Special Tag']].rename(columns={'Special Tag': 'text'})
    # Menggabungkan DataFrame
    df_combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    return df_combined

# Melakukan preprocessing text
def preprocess_text(text):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    return stopword_remover.remove(text.lower())

# NEURAL THRESHOLD untuk mendeteksi kata netral
NEURAL_THRESHOLD = 0.6

# Training model 
@st.cache_resource
def train_model(df_combined):
    # Preprocessing: Buat stopword remover dan bersihkan teks
    df_combined['cleaned_text'] = df_combined['text'].apply(preprocess_text)
    
    # Pisahkan fitur (X) dan label (y)
    X = df_combined['cleaned_text']
    y = df_combined['Sentiment']
    
    # Vectorize teks dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X)
    
    # Train model: Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vect, y)
    
    # Return trained model dan vectorizer
    return model, vectorizer

# Main page
def main():
    st.title("Sentiment Analysis Bahasa Indonesia")
    st.subheader("Streamlit Projects")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Load data
    df_combined = load_data()

    # Train model dan vectorizer
    model, vectorizer = train_model(df_combined)
    
    if choice == "Home":
        st.subheader("Home")
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
                
                # Analyze sentiment
                text_vectorized = vectorizer.transform([processed_text])  # Transform input text
                prediction = model.predict(text_vectorized)  # Membuat prediction
                sentiment_prob = model.predict_proba(text_vectorized)  # Dapatkan probabilitas
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
                token_sentiment = analyze_sentiment(processed_text, model, vectorizer)
                st.write(token_sentiment)

    else:
        st.subheader("About")

# Function untuk mengconvert sentiment ke dalam dataframe
def convert_to_df(sentiment):
    sentiment_dict = {'Sentiment': sentiment}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Function untuk menganalyze sentiment (mengelompokkan jenis kata)
def analyze_sentiment(docx, model, vectorizer):
    # Tokenisasi teks menjadi kata-kata
    tokens = docx.split()
    
    # Membuat list untuk menyimpan kata berdasarkan sentimen
    positive_words = []
    negative_words = []
    neutral_words = []
    
    for token in tokens:
        cleaned_token = token.lower().strip('.,!?"\'')  # Menghapus tanda baca dan mengubah menjadi huruf kecil
        
        # Cek apakah token ada dalam DataFrame
        token_vectorized = vectorizer.transform([cleaned_token])  # Vectorize the token
        sentiment_prob = model.predict_proba(token_vectorized)  # Dapatkan probabilitas
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
