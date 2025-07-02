import pandas as pd
import pytest
import numpy as np
from src.modeling import MODELING
from io import StringIO

@pytest.fixture
def model():
    return MODELING

@pytest.fixture
def data_preprocessed():
    data = """text,target
    i absolutely loved the new design it is fantastic,0
    the product was terrible and broke on the first day,1
    this service exceeded my expectations by far,1
    worst experience ever i will never come back,1
    """
    df = pd.read_csv(StringIO(data))
    print(df)
    return df

def test_entrainement_pipeline(model, data_preprocessed):
      
    model = MODELING()
    corpus = data_preprocessed['text']
    y = data_preprocessed['target']
    #test_size=0.3
    #random_state=42

    pipeline, X_test, y_test, y_pred = model.entrainement_pipeline(corpus, y)

    # Vérifie que le pipeline est bien entraîné
    assert pipeline is not None, "Le pipeline n'a pas été entraîné"

    # Vérifie que des prédictions sont faites
    assert y_pred is not None, "Aucune prédiction n'a été produite"

def  test_forme_prediction(model,data_preprocessed):
    corpus = data_preprocessed['text']
    y = data_preprocessed['target']
    
    model = MODELING()
    pipeline, X_test, y_test, y_pred = model.entrainement_pipeline(corpus, y)

    # Vérifie que la taille des prédictions correspond à celle du test
    assert len(y_pred) == len(X_test), "Le nombre de prédictions ne correspond pas au nombre d'échantillons de test"

    # Vérifie que les valeurs prédites sont dans les classes attendues
    assert set(np.unique(y_pred)).issubset({0, 1}), "Les classes prédites sont invalides"

    assert isinstance(len(y_pred), int), "La sortie n'est pas de type entier"

def test_metriques_retournees(model,data_preprocessed):
    model = MODELING()
    corpus = data_preprocessed['text']
    y = data_preprocessed['target']

    metriques = model.metriques_retournees(corpus, y)

    assert all(m in metriques for m in ["accuracy", "f1_score", "precision_score", "recall_score"]), "Certaines métriques sont manquantes"
    assert 0 <= metriques["accuracy"] <= 1, "Accuracy non valide"
    assert 0 <= metriques["f1_score"] <= 1, "F1-score non valide"
    assert 0 <= metriques["precision_score"] <= 1, "precision_score non valide"
    assert 0 <= metriques["recall_score"] <= 1, "recall_score non valide"

def test_comportement_text(model, data_preprocessed):
    corpus = data_preprocessed['text']
    y = data_preprocessed['target'].tolist()

    #edge_texts = ["", "a", "hi", "ok","r#"]
    model = MODELING()
    model.entrainement_pipeline(corpus, y)  # entraîne avant de prédire

    predictions = model.comportement_text(corpus)

    assert len(predictions) == len(corpus), "Nombre de prédictions incorrect pour cas limites"

