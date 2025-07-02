'''
Fonction clean_text :
Nettoie correctement texte vide, ponctuation, chiffres, mots courts
Vérification que tous les tokens ont plus de 2 lettres
Vérification que les stopwords sont supprimés
Test que le vocabulaire diminue bien après nettoyage
Vérification de l’impact du stemming/lemmatisation'''

import pandas as pd
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import pytest
from src.preprocessing import PREPROCESSING

@pytest.fixture
def preprocessing():
    return PREPROCESSING()

# Le message d'erreur s'affiche unquement s'il y a une erreur
def test_clean_text(preprocessing):
    # Données de test simples
    data_test = pd.DataFrame({
        'text': [
            "I ABSOLUTELY loved the NEW design!!! :-)",
            "Totally disappointed... the item arrived LATE and damaged. >:(",
            "I am THRILLED with the customer support!!! They were AMAZING!!!"
        ],
        'target': [0, 1, 0]
    })
      # Instanciation du préprocesseur avec le DataFrame
    preprocessing = PREPROCESSING(data=data_test)
    
    data = preprocessing.chargement_donnees()
    text = data['text']

    # Appel de la fonction à tester
    stems, supp_tokens = preprocessing.tokeniser_tweet(text)

    # Préparations
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Vérifie que la sortie n'est pas vide
    assert len(stems) > 0, "La liste des tokens est vide alors que le tweet est valide"

    # Boucle sur chaque phrase
    for phrase_stemmed, phrase_non_stemmed in zip(stems, supp_tokens):

        for stemmed_token, original_token in zip(phrase_stemmed, phrase_non_stemmed):

            # Vérifie qu'il n'y a que des lettres minuscules
            assert re.match(r"[a-zà-ÿ]+", stemmed_token), \
                f"Le token '{stemmed_token}' contient des caractères non valides"

            # Vérifie que les stopwords sont supprimés
            assert original_token not in stop_words, \
                f"Le stopword '{original_token}' est encore présent"

            # Vérifie que le stemming a bien été appliqué
            assert stemmed_token == stemmer.stem(original_token), \
                f"Le stemming de '{original_token}' aurait dû donner '{stemmed_token}'"

            # Vérifie que les mots courts (< 3 lettres) ont été supprimés
            assert len(original_token) >= 3, \
                f"Le mot court '{original_token}' n'a pas été supprimé"

def test_reforme_dataframe():
    # Données de test simples
    data_test = pd.DataFrame({
        'text': [
            "I love coding!",
            "Disaster in the city...",
            "Great service, thank you!"
        ],
        'target': [0, 1, 0]
    })

    # Crée une instance avec le DataFrame
    preprocess = PREPROCESSING(data=data_test)

    # Appelle la méthode à tester
    df_clean = preprocess.reforme_dataframe()

    # Vérifie la forme du DataFrame
    assert isinstance(df_clean, pd.DataFrame), "La sortie n'est pas un DataFrame"
    assert list(df_clean.columns) == ["text", "target"], "Les colonnes ne sont pas correctes"
    assert len(df_clean) == len(data_test), "Le nombre de lignes n'est pas respecté"
    assert df_clean['text'].apply(lambda x: isinstance(x, str)).all(), "Tous les textes ne sont pas des chaînes"
    assert df_clean['text'].str.strip().str.len().gt(0).all(), "Certains textes sont vides"
