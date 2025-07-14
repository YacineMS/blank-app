import streamlit as st

st.title(
    "Modelisation"
)

st.write("La cible du projet est d’offrir à un individu, \
    ne bénéficiant pas de l’expertise pour reconnaître une plante et sa maladie éventuelle,  \
    la capacité de l’identifier via l’appareil photo de son smartphone.  \
    Il s’agit donc d’un problème de classification des espèces ainsi que des maladies  \
    associées sur la base des images qui seront soumises par l’utilisateur. ")

col1, col2 = st.columns(2)

with col1:
    option = st.selectbox(
        "Selectionnez un model :",
        ("Keras - ResNet50V2", "Keras - EfficientNetB0", "Keras - EfficientNetV2M",
        "FastAI  - EfficientNetB0", "Tensorflow  - VGG16", "Torch - MobileNetV2", "AlexNet")
    )
    
    
with col2:
    show_subtext = st.checkbox("Voir les résultats après fine tuning")

#Affichage Finetuning

if show_subtext:
    st.write("Voici un sous-texte qui apparaît lorsque la case est cochée !")

#Selection du model
if option == "Keras - ResNet50V2":
    st.write("Vous avez sélectionné l'Option 1. Voici le texte pour l'Option 1.")
    
elif option == "Keras - EfficientNetB0":
    st.write("Vous avez sélectionné l'Option 2. Voici le texte pour l'Option 2.")
    
elif option == "Keras - EfficientNetV2M":
    st.write("Vous avez sélectionné l'Option 3. Voici le texte pour l'Option 3.")

elif option == "FastAI  - EfficientNetB0":
    st.write("Vous avez sélectionné l'Option 3. Voici le texte pour l'Option 3.")
    
elif option == "Tensorflow  - VGG16":
    st.write("Vous avez sélectionné l'Option 3. Voici le texte pour l'Option 3.")
    
elif option == "Torch - MobileNetV2":
    st.write("Vous avez sélectionné l'Option 3. Voici le texte pour l'Option 3.")
    
elif option == "AlexNet":
    st.write("Vous avez sélectionné l'Option 3. Voici le texte pour l'Option 3.")

