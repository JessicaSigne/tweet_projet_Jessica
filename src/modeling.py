
import pandas as pd
import numpy as np
#from notebooks.prog_02_Preprocessing_Modeling import Word2VecVectorizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Transformer Word2Vec compatible sklearn
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=3, min_count=1, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]  # tokenisation basique
        self.w2v_model = Word2Vec(sentences,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  sg=self.sg)
        return self

    def transform(self, X):
        sentences = [sentence.split() for sentence in X]
        X_vect = []
        for sentence in sentences:
            vectors = [self.w2v_model.wv[word] for word in sentence if word in self.w2v_model.wv]
            if vectors:
                vect_mean = np.mean(vectors, axis=0)
            else:
                vect_mean = np.zeros(self.vector_size)
            X_vect.append(vect_mean)
        return np.array(X_vect)


class MODELING:
    def __init__(self):
        self.pipeline = None
    
    def entrainement_pipeline(self, corpus, y):
        
        X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42)

        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer()),
                ('w2v', Word2VecVectorizer())
            ])),
            ('clf', LogisticRegression(max_iter=500))
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        return self.pipeline, X_test, y_test, y_pred

    def forme_prediction(self, corpus, y):
        self.pipeline, X_test, y_test, y_pred = self.entrainement_pipeline(corpus, y)
        return len(X_test), len(y_pred)

    def metriques_retournees(self, corpus, y):
        self.pipeline, X_test, y_test, y_pred = self.entrainement_pipeline(corpus, y)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        return {"accuracy": acc, "f1_score": f1, "precision_score":prec, "recall_score": rec}

    def comportement_text(self, edge_texts):
        if not hasattr(self, 'pipeline'):
            raise ValueError("Le pipeline n'est pas entraîné. Appelez `entrainement_pipeline` d'abord.")
        return self.pipeline.predict(edge_texts)

    
    def afficher_metriques(self, corpus, y):
        print("Entraînement terminé.")
        print("MÉTRIQUES DU MODÈLE".center(40))

        # Entraînement du modèle pour récupérer pipeline et prédictions
        self.pipeline, X_test, y_test, y_pred = self.entrainement_pipeline(corpus, y)

        # Affichage des métriques
        metriques = self.metriques_retournees(corpus, y)
        print("=" * 40)
        for k, v in metriques.items():
            print(f"{k.upper():<20} : {v:.4f}")

        # Affichage d'exemples de prédictions
        print("=" * 40)
        print(f"Nombre d'exemples testés : {len(X_test)}")
        print("\nQUELQUES EXEMPLES DE PRÉDICTIONS".center(40))
        print("-" * 40)
        for phrase, pred in zip(X_test[:3], y_pred[:3]):
            print(f"Texte : {phrase[:60].strip():<60} → Prédit : {pred}")


