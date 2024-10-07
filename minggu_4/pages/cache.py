import streamlit as st
import pandas as pd
import time

# Data dummy dibuat secara manual menggunakan array (list of dictionaries)
data_dummy = [
    {"date": "2024-09-01", "rides": 50, "distance": 120.5},
    {"date": "2024-09-02", "rides": 75, "distance": 150.0},
    {"date": "2024-09-03", "rides": 100, "distance": 175.3},
    {"date": "2024-09-04", "rides": 200, "distance": 250.2}
]

# Dekorator cache untuk menyimpan hasil proses
@st.cache_data
def load_data(data):
    # Simulasi delay untuk loading
    time.sleep(2)  # Tambahkan sedikit waktu loading untuk simulasi
    df = pd.DataFrame(data)
    return df

# Menambahkan indikator loading saat proses caching
with st.spinner('Loading data...'):
    df = load_data(data_dummy)

# Menampilkan DataFrame di Streamlit setelah loading selesai
st.dataframe(df)

# Tombol untuk rerun aplikasi dan clear cache
if st.button("Rerun and Clear Cache"):
    # Mererun aplikasi dan membersihkan cache
    st.cache_data.clear()  # Membersihkan cache secara manual
