import streamlit as st
import pandas as pd

def show_ensemblage() :
    st.title(
        "Ensemblage des modèles "
    )
    st.write('''
            Nous étudions ici l’impact de l’ensemblage de modèles sur 
            l’amélioration des performances de classification, notamment sur la class Cassava.
        ''')

    st.subheader("Performance individuelle des modèles ")
    st.image('/workspaces/blank-app/ressources/model_architecture/performance ind ensemblage.png', caption='Correlation des erreurs')

    col1, col2 = st.columns(2)
    with col1 :
        st.subheader("Corrélation des erreurs")
        st.image('/workspaces/blank-app/ressources/model_resultat/correlation_erreur.png', caption='Corrélation des erreurs')

    with col2:
        st.subheader("Observations")
        st.write('''
            ResNet50V2 est le modèle le moins corrélé sur les erreurs avec les autres modèles. Cela indique une bonne complémentarité avec les autres modèles. 
            EfficientNetV2M ,ConvNeXt et SwinTransformer sont les modèles les plus relativement corrélés entre eux sur les erreurs.
            Cependant le niveau de corrélation reste suffisamment bas (inférieur à 0.51 ) pour justifier un ensemblage. \n
            Ces résultats confirment l’intérêt d’un ensemblage basé sur ces modèles, car la diversité des erreurs est un facteur favorable à l'amélioration globale des performances 
        ''')
    col3, col4 = st.columns(2)
    with col3 :
        st.subheader("Divergence de Jensen-Shannon")
        st.image('/workspaces/blank-app/ressources/model_resultat/correlation_jensen_shannon.png', caption='Divergence de Jensen-Shannon')

    with col4:
        st.subheader("Observations")
        st.write('''
            ResNet50V2 présente la plus forte divergence avec tous les autres modèles ce qui confirme son apport en termes de diversité dans les ensembles. 
            ConvNext, EfficientNetV2M et Swintransformer ont une diversité modérée entre eux, ce qui indique une certaine redondance dans leurs décisions. 
        ''')
        
    st.subheader("Résultat de l’ensemblage")
    st.write('''
            La technique utilisé pour l'ensemblage est le soft voting, dans lequel chaque modèle contribue à la prédiction finale via sa distribution de probabilité. 
    ''')
    
    data = {
        "rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "models": [
            "EffNetV2M + ResNet50V2 + Swin",
            "EffNetV2M + ResNet50V2",
            "EffNetV2M",
            "EffNetV2M + ConvNeXt + ResNet50V2",
            "EffNetV2M + Swin",
            "EffNetV2M + ConvNeXt + Swin",
            "EffNetV2M + ConvNeXt",
            "ResNet50V2 + Swin",
            "ConvNeXt + ResNet50V2 + Swin",
            "ConvNeXt + Swin",
            "Swin",
            "ConvNeXt + ResNet50V2",
            "ConvNeXt",
            "ResNet50V2"
        ],
        "accuracy": [
            0.9865, 0.9855, 0.9858, 0.9841, 0.9851, 0.9846, 0.9839, 0.9808, 0.9801, 0.9801, 0.9786, 0.9741, 0.9697, 0.9578
        ],
        "f1_score": [
            0.9856, 0.9847, 0.9845, 0.9837, 0.9834, 0.9833, 0.9828, 0.9796, 0.9792, 0.9789, 0.9763, 0.9740, 0.9677, 0.9581
        ],
        "nb_models": [
            3, 2, 1, 3, 2, 3, 2, 2, 3, 2, 1, 2, 1, 1
        ]
    }

    df = pd.DataFrame(data)
    st.dataframe(df)
    
    st.subheader("Observations et conclusion")
    st.write('''
    Globalement, l’ensemble EfficientNetV2M + ResNet50V2 + SwinTransformer affiche les meilleures performances, avec: \n
        ●  Accuracy : 0.9865 
        ●  F1-score : 0.9856 
    
    Dans l’application l’ensemble à deux modèles CNN: EfficientNetV2M + ResNet50V2 qui est utilisé systématiquement. 
    SwinTranformer est utilisé en renfort sur les cas difficiles grâce à son architecture type transformer. 
''')