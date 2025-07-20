import streamlit as st
import pandas as pd
from modelisation import show_modelisation
from ensemblage import show_ensemblage


page = st.sidebar.selectbox("Choisissez une page", ["Modelisation", "Ensemblage", "Interpretation"])

if page == "Modelisation":
    show_modelisation()

elif page == "Ensemblage":
    show_ensemblage()