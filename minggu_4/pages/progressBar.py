import streamlit as st 
import time

st.markdown('# Progress Bar')
#Progress Bar
'Starting a long computation...'
#Menambahkan Placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    #Update the progress bar with each iteartion
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)