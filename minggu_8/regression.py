# Core Pkgs
import streamlit as st
import sklearn
import joblib, os
import numpy as np 

# Loading Models
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Sales Prediction Berdasarkan Iklan di TV, Radio, dan Newspaper"""

    st.title("Prediksi Penjualan Berdasarkan Iklan")

    html_templ = """
    <div style="background-color:#726E6D;padding:10px;">
    <h3 style="color:white">Prediksi Penjualan Menggunakan Regresi Linier</h3>
    </div>
    """

    st.markdown(html_templ, unsafe_allow_html=True)

    activity = ["Prediksi Penjualan", "Apa itu Regresi?"]
    choice = st.sidebar.selectbox("Menu", activity)

    # Sales Prediction CHOICE
    if choice == 'Prediksi Penjualan':

        st.subheader("Prediksi Penjualan")

        # Input dari pengguna: Jumlah iklan di TV, Radio, dan Newspaper
        tv_ad = st.number_input("Berapa jumlah pengeluaran iklan di TV? (dalam ribuan dolar)", min_value=0.0)
        radio_ad = st.number_input("Berapa jumlah pengeluaran iklan di Radio? (dalam ribuan dolar)", min_value=0.0)
        newspaper_ad = st.number_input("Berapa jumlah pengeluaran iklan di Newspaper? (dalam ribuan dolar)", min_value=0.0)

        if st.button("Prediksi Penjualan"):

            # Load the trained model
            regressor = load_prediction_model("Minggu_8/sales_prediction_model.pkl")

            # Reshape input to match the model's expected input
            input_data = np.array([tv_ad, radio_ad, newspaper_ad]).reshape(1, -1)

            # Predict sales based on input
            predicted_sales = regressor.predict(input_data)

            st.info("Prediksi Penjualan untuk pengeluaran iklan di TV: ${}, Radio: ${}, dan Newspaper: ${} adalah: {:.2f} unit penjualan".format(tv_ad, radio_ad, newspaper_ad, predicted_sales[0]))

    elif choice == "Apa itu Regresi?":
        st.subheader("Apa itu Regresi Linier?")
        st.write("""
        Regresi linier adalah metode statistik yang digunakan untuk memodelkan hubungan antara variabel independen (input) dan variabel dependen (output) dengan menggunakan garis lurus. Dalam kasus ini, kami mencoba memprediksi jumlah penjualan berdasarkan pengeluaran iklan di TV, Radio, dan Koran.
        """)

if __name__ == '__main__':
    main()
