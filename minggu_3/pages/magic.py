import streamlit as st
import pandas as pd

st.markdown("# Magic Data Frame")
df = pd.DataFrame({
    'first column' : [5,4,3,2,1],
    'second column' : [10,20,30,40,50]
})

df

# Data untuk stok barang mini market
data = {
    'Nama Barang': ['Beras', 'Gula', 'Minyak Goreng', 'Susu', 'Telur'],
    'Kategori': ['Makanan Pokok', 'Makanan Pokok', 'Makanan Pokok', 'Minuman', 'Makanan Pokok'],
    'Stok': [50, 75, 40, 30, 120],
    'Harga (IDR)': [10000, 12000, 15000, 8000, 2000]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan DataFrame menggunakan Streamlit
st.title('Stok Barang Mini Market')
st.dataframe(df)