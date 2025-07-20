import streamlit as st
import pandas as pd

def show_interpretation() :
    st.title(
        "Interpretation"
    )

    st.write('''
        Pour comprendre les décisions de nos, nous avons utilisé 4 outils d’interprétation :
        - Saliency, qui permet de mettre en évidence les pixels les plus importants d’une image pour une prédiction donnée.
        - SHAP (SHapley Additive exPlanations), pour visualiser les caractéristiques qui permettent la prédictions d’une classe sur une image.
        - LIME (Local Interpretable Model-agnostic Explanations), qui donne les caractéristiques d’une image qui vont le plus influencer la décision du model.
        - GradCam qui permet de visualiser les partie d'une image qui ont été les plus importantes pour le modèle
    ''')

    st.image('/workspaces/blank-app/ressources/interpretation/interpretation_saillence_shap_LIME.png', caption='Interpretation du model')

    st.image('/workspaces/blank-app/ressources/interpretation/gradcam.png', caption='Interpretation du model')
