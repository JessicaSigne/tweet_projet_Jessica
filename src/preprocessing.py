'''
Fonction 
clean_text
 :
 Nettoie correctement texte vide, ponctuation, chiffres, mots courts
 Vérification que tous les tokens ont plus de 2 lettres
 Vérification que les stopwords sont supprimés
 Test que le vocabulaire diminue bien après nettoyage
 Vérification de l’impact du stemming/lemmatisation'''

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

class PREPROCESSING:
    def __init__(self, chemin=None, data=None):
        self.chemin = chemin
        self.data = data  # stocke le DataFrame s'il est passé directement

    def chargement_donnees(self):
        if self.data is not None:
            return self.data  # On utilise les données déjà nettoyées
        elif self.chemin is not None:
            try:
                data = pd.read_csv(self.chemin, sep=',')
                return data
            except FileNotFoundError:
                raise FileNotFoundError(f"Fichier non trouvé : {self.chemin}")
        else:
            raise ValueError("Veuillez fournir un chemin ou un DataFrame.")


    def tokeniser_tweet(self, text = None):
        data = self.chargement_donnees()
        text =  data['text'].dropna().tolist()

        # Étape 1 : nettoyage de chaque phrase (minuscule, suppression des caractères non alphabétiques)
        text_clean = [re.sub(r"http\S+|[^a-zA-ZÀ-ÿ\s]", "", phrase.lower()) for phrase in text if isinstance(phrase, str)]

        # Étape 2 : suppression des mots de moins de 3 lettres, phrase par phrase
        text_clean = [' '.join([word for word in phrase.split() if len(word) >= 3]) for phrase in text_clean]

        # TOKENISATION
        tokens=  [word_tokenize(phrase) for phrase in text_clean]

        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        supp_tokens = []
        stems = []

        for phrase in text_clean:
            tokens = phrase.split()  # tokenisation simple, phrase par phrase
            filtered = [word for word in tokens if word.lower() not in stop_words]
            supp_tokens.append(filtered)
            stems.append([stemmer.stem(word) for word in filtered])
        
        return stems, supp_tokens
    
    def reforme_dataframe(self):
        stems, supp_tokens = self.tokeniser_tweet()
        data = self.chargement_donnees()

        texts = []
        for words in stems:
            texts.append(' '.join(words))

        data['text'] = texts
        data = data[['text', 'target']]
        print(data)
        return data