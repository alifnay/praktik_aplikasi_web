import streamlit as st

# Inisialisasi session state untuk menghitung jumlah klik
if 'count' not in st.session_state:
    st.session_state.count = 0

# Fungsi untuk menambah counter ketika tombol ditekan
def increment_counter():
    st.session_state.count += 1

# Tampilkan nilai counter saat ini
st.write(f"Button clicked: {st.session_state.count} times")

# Tombol untuk menambah counter
st.button("Click me!", on_click=increment_counter)

# Tombol untuk mereset counter
if st.button("Reset"):
    st.session_state.count = 0