import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building!"
)

# Case à cocher pour afficher un sous-texte
if st.checkbox("Afficher le sous-texte"):
    st.write("Voici un sous-texte qui apparaît lorsque la case est cochée !")