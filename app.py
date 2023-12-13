import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from analyst_model import model_page


page = st.sidebar.selectbox("Explore Or Predict or Analyze Model", ("Predict", "Explore","Analyze Model"))

if page == "Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()
elif page == "Analyze Model":
    model_page() 