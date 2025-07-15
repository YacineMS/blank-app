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


#Selection du model
if option == "Keras - ResNet50V2":
    st.markdown("<u>Keras</u>", unsafe_allow_html=True)
    st.write('''
             Keras est une bibliothèque open source de réseaux de neurones écrite en Python. Avec un haut niveau
             d\'abstraction, sa simplicité et sa modularité, elle est  l\’une des APIs de réseaux de neurones les plus utilisées.
            ''')
    st.write('''
             ResNet50V2 est une version améliorée de l'architecture de réseau de neurones ResNet50, 
             conçue pour la vision par ordinateur. Il utilise des blocs résiduels, qui permettent de mieux entraîner des réseaux profonds 
             en atténuant le problème de dégradation de la performance
             ''')
    st.write("Vous avez sélectionné l'Option 1. Voici le texte pour l'Option 1.")
    
    #Affichage Finetuning
    if show_subtext:
        st.write("Voici un sous-texte qui apparaît lorsque la case est cochée !")
        
elif option == "Keras - EfficientNetB0":
    st.markdown("<u>Keras</u>", unsafe_allow_html=True)
    st.write('''
             Keras est une bibliothèque open source de réseaux de neurones écrite en Python. Avec un haut niveau
             d\'abstraction, sa simplicité et sa modularité, elle est  l\’une des APIs de réseaux de neurones les plus utilisées.
            ''')
    st.write('''
            EfficientNet utilise une méthode d'optimisation appelée "compound scaling", qui ajuste simultanément la profondeur, 
            la largeur et la résolution de l'image, offrant ainsi une meilleure efficacité énergétique et une précision supérieure.
            Il est le plus petit des modèles de la famille EfficientNet, ce qui le rend pertinent pour 
            son utilisation sur du matériel avec des ressources limitées (smartphone, tablette).
        ''')
    
elif option == "Keras - EfficientNetV2M":
    st.markdown("<u>Keras</u>", unsafe_allow_html=True)
    st.write('''
             Keras est une bibliothèque open source de réseaux de neurones écrite en Python. Avec un haut niveau
             d\'abstraction, sa simplicité et sa modularité, elle est  l\’une des APIs de réseaux de neurones les plus utilisées.
            ''')
    st.write('''
                Une amélioration de l'EfficientNet original, qui intègre des techniques de fusion des convolutions et un meilleur équilibrage des ressources.
                Plus complexe et plus profond qu'EfficientNetB0, il est très performant sur des tâches complexes et des jeux de données volumineux.
            ''')

elif option == "FastAI  - EfficientNetB0":
    st.markdown("<u>FastAI</u>", unsafe_allow_html=True)
    st.write('''
                S\'appuyant sur PyTorch, fast.ai est un framework conçu pour rendre l'apprentissage profond accessible au plus grand nombre.
                Il se concentre sur la simplicité et la rapidité de mise en œuvre avec un très haut niveau d'abstraction.
             ''')
    st.write('''
                EfficientNet utilise une méthode d'optimisation appelée "compound scaling", qui ajuste simultanément la profondeur, 
                la largeur et la résolution de l'image, offrant ainsi une meilleure efficacité énergétique et une précision supérieure.
                Il est le plus petit des modèles de la famille EfficientNet, ce qui le rend pertinent pour 
                son utilisation sur du matériel avec des ressources limitées (smartphone, tablette).
            ''')    
elif option == "Tensorflow  - VGG16":
    st.markdown("<u>Tensorflow</u>", unsafe_allow_html=True)
    st.write('''
                Développé par Google, Tensorflow est un framework qui offre une grande flexibilité, 
                largement utilisé dans la recherche et en production. Il est apprécié pour sa capacité à gérer des modèles complexes et 
                à les déployer à grande échelle.
            ''')

    st.markdown("<u>Modèle</u>", unsafe_allow_html=True)
    st.write('''
                VGG16 est une architecture de réseau de neurones convolutifs (CNN), devenue populaire par sa simplicité et son utilisation de convolutions 3x3, 
                ce qui en fait l'une des architectures les plus influentes dans le domaine de la vision par ordinateur.
            ''')
elif option == "Torch - MobileNetV2":
    st.markdown("<u>Torch</u>", unsafe_allow_html=True)
    st.write('''
             Grâce à ses bibliothèques et modules spécialisés, Torch a été largement utilisé pour des tâches de vision par ordinateur.
             Ses concepts et ses bibliothèques comme le calcul sur tenseurs ou le support pour des calculs GPU ont jeté les bases pour des frameworks plus modernes comme PyTorch.
             ''')

    st.markdown("<u>Modèle</u>", unsafe_allow_html=True)
    st.write('''
                Particulièrement adaptée pour les tâches de vision par ordinateur sur des appareils avec des ressources limitées,
                MobileNetV2 possède une architecture légère et efficace, composé de blocs inversés résiduels
                qui permettent de réduire considérablement le nombre de paramètres et les opérations nécessaires, tout en maintenant une bonne précision.
            ''')
elif option == "AlexNet":
    st.write('''
                AlexNet est une architecture de réseau de neurones convolutifs (CNN) utilisant des couches convolutives et des unités de rectification linéaire (ReLU). 
                Il a significativement amélioré la précision de la reconnaissance d'images sur des jeux de données complexes comme ImageNet.
            ''')
