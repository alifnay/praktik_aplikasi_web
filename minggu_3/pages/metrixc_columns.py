import streamlit as st

st.markdown("# Metric")
st.metric(label="Harga Saham Pertamina", value=1.185, delta= 1)
st.metric(label="Harga Saham GoTo", value=950, delta= -0.5)
st.metric(label="Total Saham", value=1122000, delta= 100, delta_color="off")
st.markdown("---")

st.markdown("# Columns")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", "25 °C", "1.2 °C")
col2.metric("Wind", "12 kph", "-2%")
col3.metric("Humidity", "89%", "3%")
col4.metric("Rain", "1.6mm")