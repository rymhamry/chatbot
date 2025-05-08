import nltk

# Spécifie le dossier où les données doivent être téléchargées
chemin = r"C:\Users\HP\nltk_data"
nltk.download('punkt', download_dir=chemin)
nltk.download('stopwords', download_dir=chemin)

# Ajoute ce dossier dans le chemin de recherche de NLTK
nltk.data.path.append(chemin)
