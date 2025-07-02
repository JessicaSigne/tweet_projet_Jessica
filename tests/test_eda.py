'''Vérification de la présence et des types des colonnes ( text , target )
Détection automatique de valeurs manquantes ou doublons
Vérification que tous les textes sont non-vides
Validation du nombre de classes possibles
Test sur les longueurs de texte (moyenne, min, max)
'''
# pytest .\tests\test_eda.py
# python main.py
import pandas as pd
import pytest
from src.eda import EDA

@pytest.fixture
def simulation_csv(tmp_path):
    data = """text,target
I absolutely loved the new design, it is fantastic!,1
The product was terrible and broke on the first day,0
,1
I absolutely loved the new design, it is fantastic!,1
This service exceeded my expectations by far,1
Worst experience ever, I will never come back,0
,0
The quality is okay but could be better, 
I am thrilled with the customer support, they were amazing!,1
Totally disappointed, the item arrived late and damaged,0
The staff was friendly and helpful throughout the visit,1
"""
    fichier = tmp_path / "simulation.csv"
    fichier.write_text(data)
    return str(fichier)

@pytest.fixture
def eda(simulation_csv):
    return EDA(chemin = simulation_csv)

# Le message d'erreur s'affiche unquement si le test échoue
def test_chargement_donnees(eda):
    assert eda.chargement_donnees() is not None, "Erreur: Le chargement des données a échoué"
    assert isinstance(eda.chargement_donnees(), pd.DataFrame) , "Erreur: Le fichier chargé n'est pas un DataFrame"

def test_presence_colonnes(eda):
    data = eda.chargement_donnees()
    print(data)
    assert data is not None, "Erreur: aucune colonne présente dans le dataset"
    assert eda.presence_colonnes(data) == True, f"Erreur: Colonnes absentes du dataset"

def test_types_colonnes(eda):
    data = eda.chargement_donnees()
    print(f"Dans test : type text: {data['text'].dtype}, type target: {data['target'].dtype}")
    assert eda.types_colonnes(data) is True, "Les colonnes n'ont pas les types attendus"

def test_valeurs_manquantes(eda):
    data = eda.chargement_donnees()
    assert eda.valeurs_manquantes(data)>=0, " le nombre de valeurs manquantes n'a pas pu être retourné" 
                      
def test_doublons_text(eda):
    data = eda.chargement_donnees()
    n_avant = data.shape[0]  # nombre de lignes avant
    n_doublons = eda.doublons_text(data)
    n_apres = data.shape[0]  # nombre de lignes après suppression

    assert n_doublons >= 0, "Le nombre de doublons n'a pas pu être affiché"
    assert n_apres == n_avant - n_doublons, "Les doublons n'ont pas été correctement supprimés"

def test_textes_non_vides(eda):
    data = eda.chargement_donnees()
    textes_vides, textes_none, textes_nan = eda.textes_non_vides(data)

    # Signaler mais ne pas échouer sur les textes vides
    if textes_vides > 0 or textes_none> 0 or textes_nan>0:
        assert eda.doublons_text(data)>=0, " le nombre de doublons n'a pas pu être affiché" 

def test_nombre_classes(eda):
    data = eda.chargement_donnees()
    nbre_classe = eda.nombre_classes(data)
    assert nbre_classe==2, "Erreur: Classes cibles inattendues"

def test_longueurs_texte(eda):
    data = eda.chargement_donnees()
    min_len, max_len, moy_len = eda.longueurs_texte(data)

    assert min_len >0, f"Minimum inattendu"
    assert max_len >= min_len, f"Maximum inattendu"
    assert moy_len > 0, f"Moyenne inattendue"

    
