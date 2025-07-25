import streamlit as st
import pandas as pd
from modelisation import show_modelisation
from ensemblage import show_ensemblage
from interpretation import show_interpretation


page = st.sidebar.selectbox("Choisissez une page", ["Modelisation", "Interpretation", "Ensemblage"])

if page == "Modelisation":
    show_modelisation()
    
elif page == "Interpretation":
    show_interpretation()
    
elif page == "Ensemblage":
    show_ensemblage()
