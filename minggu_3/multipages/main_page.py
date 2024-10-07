import streamlit as st

def main_page():
    st.markdown("# Main Page 🎈")
    st.sidebar.markdown("# Main Page 🎈")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

pages_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3
}

selected_page = st.sidebar.selectbox("Select a page" ,pages_names_to_funcs.keys())
pages_names_to_funcs[selected_page]()
