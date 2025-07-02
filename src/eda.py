'''Vérification de la présence et des types des colonnes ( text , target )
Détection automatique de valeurs manquantes ou doublons
Vérification que tous les textes sont non-vides
Validation du nombre de classes possibles
Test sur les longueurs de texte (moyenne, min, max)
'''

import pandas as pd

class EDA:
    def __init__(self, chemin='chemin'):
        self.chemin = chemin

    def chargement_donnees(self):
        try:
            data = pd.read_csv(self.chemin, sep=',', dtype={"text": object, "target": "Int64"})
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier non trouvé : {self.chemin}")
        except ValueError as e:
            # Gérer d'autres erreurs possibles de dtype
            raise ValueError(f"Erreur lors du chargement des données avec dtype forcé : {e}")

    def presence_colonnes(self, data):
        colonnes_attendues = ['text', 'target']
        colonnes_absentes = [col for col in colonnes_attendues if col not in data.columns]
        if colonnes_absentes:
            raise ValueError(f"Erreur: Colonnes absentes du dataset : {colonnes_absentes}")
        return True

    def types_colonnes(self, data):
        # Vérifier si 'text' est de type object (chaine de caractères)
        text_object = data['text'].dtype == object
        
        # Vérifier si 'target' est de type int (integer)
        # Utilisation plus robuste
        target_int = pd.api.types.is_integer_dtype(data['target'])
        
        # Optionnel : afficher les types pour debug
        print(f"type text: {data['text'].dtype}, type target: {data['target'].dtype}")
        
        return text_object and target_int

    def valeurs_manquantes(self,data):
        colonnes_attendues = ['text', 'target']
        total_manquantes = sum(data[col].isna().sum() for col in colonnes_attendues if col in data.columns)
        return total_manquantes

    def doublons_text(self, data):
        doublons = data['text'].duplicated().sum()
        data.drop_duplicates(subset='text', inplace=True)
        return doublons

    def textes_non_vides(self , data):
        textes_vides = data['text'].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
        textes_none = data['text'].apply(lambda x: x is None).sum()
        textes_nan = data['text'].isna().sum()
        
        return textes_vides, textes_none, textes_nan

    def nombre_classes(self , data):
        classes_uniques = data['target'].nunique()
        return classes_uniques

    def longueurs_texte(self , data):
        longueurs = data['text'].apply(lambda x: len(str(x)))
        
        return longueurs.min(), longueurs.max(), longueurs.mean()

    def afficher_infos(self, data):
            """
            Méthode pour afficher les informations globales d'analyse exploratoire.
            """
            # Vérifications de base
            self.presence_colonnes(data)
            assert self.types_colonnes(data), "Problème de types dans les colonnes"
            print("Types de colonnes OK")

            print(f"Valeurs manquantes : {self.valeurs_manquantes(data)}")
            print(f"Doublons supprimés : {self.doublons_text(data)}")
            print(f"Textes vides / None / NaN : {self.textes_non_vides(data)}")
            print(f"Nombre de classes : {self.nombre_classes(data)}")

            min_len, max_len, avg_len = self.longueurs_texte(data)
            print(f"Longueurs des textes → min: {min_len}, max: {max_len}, moyenne: {avg_len:.2f}")
