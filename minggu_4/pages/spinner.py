import time
import streamlit as st

st.markdown('# Spinner')
with st.spinner('Wait for it...'):
    time.sleep(5)
st.success("Done!")