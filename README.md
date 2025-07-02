# Projet de Classification de Tweets — Jessica

Bienvenue dans le projet **Tweet Project** réalisé dans le cadre de mon TP test unitaire en Master 1 Data Science.  
L'objectif principal est de **classer automatiquement des tweets** en deux catégories : `catastrophes` ou `normaux` à l’aide de techniques de machine learning et de traitement du langage naturel (NLP).


## Objectif du projet 
```
- Faire de l'analyse de données sur un dataframe comprenant les tweets 
- Nettoyer le corpus de textes issus de tweets
- Appliquer des techniques de prétraitement NLP adaptées
- Construire une pipeline avec un modèle de classification(LogisticRegression) et une méthode de vectorisation (Word2Vec)
- Évaluer la qualité du modèle avec des métriques standard
- Offrir une compatibilité Docker pour exécution simplifiée 
```

## Structure du projet
```
TWEET_PROJECT_JESSICA/
│
├── data/                          # Dossier des données brutes
│   └── tweet.csv                  # Fichier principal contenant les tweets
│
├── notebooks/                     # Notebooks Jupyter pour l’exploration et le prototypage
│   ├── __init__.py                # Permet d'importer le dossier comme un package
│   ├── prog_01_EDA.ipynb          # Analyse exploratoire des données (EDA)
│   └── prog_02_Preprocessing_Modeling.ipynb  # Nettoyage et modélisation
│
├── src/                           # Code source principal
│   ├── __init__.py                # Déclare le module src
│   ├── eda.py                     # Fonctions pour l'analyse exploratoire
│   ├── preprocessing.py           # Fonctions pour le nettoyage, la transformation et la préparation des tweets  
│   └── modeling.py                # Fonctions pour l'entraînement, l'évaluation et les prédictions du modèle
│
├── tests/                         # Tests unitaires automatisés
│   ├── __init__.py                # Déclare le module tests
│   ├── test_eda.py                # Tests pour les fonctions EDA
│   ├── test_preprocessing.py      # Tests pour le prétraitement des textes
│   └── test_modeling.py           # Tests pour la modélisation 
│
├── main.py                        # Script principal qui exécute le pipeline complet
├── Dockerfile                     # Conteneurisation du projet pour un déploiement simple
├── requirements.txt               # Liste des bibliothèques Python nécessaires
└── README.md                      # Description du projet, instructions et documentation
```

## Présentation des données principales choisies du dataset

**text** : Colonne qui contient les données textuelles sur les tweets. Les données sont en textes brutes et mal formatées.

**target** : Colonne cible qui affecte des valeurs 1 pour les tweets catastrophiques et 0 pour les normaux.

## Lancement du projet

### Exécution du script principal

#### Exécution du script principal (avec Docker)

**Construction de l’image Docker :**
```docker build -t tweet_project_jessica .```

**Exécution du script complet :**
```docker run --rm tweet_project_jessica```

**Exécution des tests uniquement :**
```docker run --rm tweet_project_jessica pytest```

### Tests automatisés

--- Les tests permettent de garantir la robustesse des étapes EDA, prétraitement et modélisation. Ils vérifient les fonctions créées pour chaque étape:

**Tests eda**

```
- Vérification de la présence et des types des colonnes ( text , target )
- Détection automatique de valeurs manquantes ou doublons
- Vérification que tous les textes sont non-vides
- Validation du nombre de classes possibles
- Test sur les longueurs de texte (moyenne, min, max)
```

**Tests preprocessing**
```
- Nettoie correctement texte vide, ponctuation, chiffres, mots courts
- Vérification que tous les tokens ont plus de 2 lettres
- Vérification que les stopwords sont supprimés
- Test que le vocabulaire diminue bien après nettoyage
- Vérification de l’impact du stemming
```

**Tests modeling**
```
- Vérifier que le pipeline s'entraîne sans erreur sur un jeu d’exemple
- Vérifier la forme des prédictions
- Vérifier que les métriques sont retournées correctement
- Comportement attendu sur texte vide, ou très court
```

## Technologies utilisées

```
Python 3.11 (Version python utilisée)

Scikit-learn (classification des tweets)

NLTK (Nettoyage des tweets)

Gensim (Word2Vec) (vectorisation des textes)

Pandas, NumPy (création et manipulation des dataframess)

Pytest (tests unitaires)

Docker (Construire l’image du projet)
```

## Réalisé par

```
Jessica SIGNE 
Étudiante en Master 1 Data Science à YNOV Campus
Contact : jessicasigne44@gmail.com
```
#### Lien vers le git
``` https://github.com/JessicaSigne/tweet_projet_Jessica/tree/master ```




