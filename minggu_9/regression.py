# Core Pkgs
import streamlit as st
import sklearn
import joblib, os
import numpy as np
import plotly.graph_objects as go

# Loading Models
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Sales Prediction Berdasarkan Iklan di TV"""

    st.title("Prediksi Hasil Penjualan Berdasarkan Iklan")

    html_templ = """
    <div style="background-color:#726E6D;padding:10px;">
    <h3 style="color:white">Prediksi Hasil Penjualan Menggunakan Regresi Linier</h3>
    </div>
    """

    st.markdown(html_templ, unsafe_allow_html=True)

    activity = ["Prediksi Hasil Penjualan", "Apa itu Regresi?"]
    choice = st.sidebar.selectbox("Menu", activity)

    # Sales Prediction CHOICE
    if choice == 'Prediksi Hasil Penjualan':

        st.subheader("Prediksi Hasil Penjualan")

        # Input dari pengguna: Jumlah iklan di TV, Radio, dan Newspaper
        tv_ad = st.number_input("Berapa jumlah pengeluaran iklan di TV? (dalam ribuan dolar)", min_value=0.0)

        if st.button("Prediksi Hasil Penjualan"):

            # Load the trained model
            regressor = load_prediction_model("minggu_9/linear_regression_tv.pkl")

            # Reshape input to match the model's expected input
            input_data = np.array([tv_ad]).reshape(1, -1)

            # Predict sales based on input
            predicted_sales = regressor.predict(input_data)
            
            # Mengakses elemen tunggal dalam array hasil prediksi
            predicted_sales_value = predicted_sales[0][0]

            st.info("Prediksi Hasil Penjualan untuk pengeluaran iklan di TV: \${} adalah: \${:.2f} Juta".format(tv_ad, predicted_sales_value))

            # Plotting with Plotly
            fig = go.Figure()

            # Adding the predicted point
            fig.add_trace(go.Scatter(
                x=[tv_ad], y=[predicted_sales_value], mode='markers',
                marker=dict(color='red', size=12),
                name='Prediksi Hasil Penjualan'
            ))

            # Update layout for better visualization
            fig.update_layout(
                title="Hubungan antara Pengeluaran Iklan TV dan Penjualan",
                xaxis_title="Pengeluaran Iklan di TV ($ dalam ribuan dolar)",
                yaxis_title="Hasil Penjualan ($ Juta dollar)",
                showlegend=True
            )

            # Display the Plotly graph
            st.plotly_chart(fig)

    elif choice == "Apa itu Regresi?":
        st.subheader("Apa itu Regresi Linier?")
        st.write("""
        Regresi linier adalah metode statistik yang digunakan untuk memodelkan hubungan antara variabel independen (input) dan variabel dependen (output) dengan menggunakan garis lurus. Dalam kasus ini, kami mencoba memprediksi jumlah penjualan berdasarkan pengeluaran iklan di TV, Radio, dan Koran.
        """)

if __name__ == '__main__':
    main()
