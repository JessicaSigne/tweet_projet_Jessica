# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier le fichier requirements.txt dans l'image
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

#RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Définir la variable d'environnement pour que NLTK sache où chercher
ENV NLTK_DATA=/usr/local/nltk_data

# Copier le reste du projet dans l'image
COPY . .

# Définir une variable d'environnement par défaut
ENV CMD="python main.py"

# Commande par défaut : exécuter ce qui est dans CMD
ENTRYPOINT ["sh", "-c"]
CMD ["$CMD"]


#Commandes à lancer pour docker dans l'ordre
#docker build -t tweet_project_jessica . : construire l'image de mon projet
#docker run --rm tweet_project_jessica : lancer l'ensemble du projet
#docker run --rm tweet_project_jessica pytest : lancer juste les tests
