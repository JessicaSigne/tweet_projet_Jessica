import os
from src.eda import EDA
from src.preprocessing import PREPROCESSING
from src.modeling import MODELING

def main():
    print("Bienvenue dans le projet Tweet de Jessica !\n")

    # Définir le chemin relatif vers le fichier
    chemin = os.path.join("data/tweet.csv")

    # Étape 1 : Analyse exploratoire (EDA)
    print("Étape 1 : Analyse des données...")
    eda = EDA(chemin=chemin)
    data = eda.chargement_donnees()
    eda.afficher_infos(data)

    # Étape 2 : Nettoyage et préparation (PREPROCESSING)
    print("\nÉtape 2 : Nettoyage des textes...")
    preprocessing = PREPROCESSING(data=data)
    data_clean = preprocessing.reforme_dataframe()

    # Étape 3 : Modélisation (MODELING)
    print("\n Étape 3 : Entraînement du modèle...")
    modeling = MODELING()
    pipeline, X_test, y_test, y_pred = modeling.entrainement_pipeline(data_clean['text'], data_clean['target'])
    modeling.afficher_metriques(data_clean['text'], data_clean['target'])


if __name__ == "__main__":
    main()



