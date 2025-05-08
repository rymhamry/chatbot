import nltk
import os

# Définir un dossier local pour les ressources NLTK
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Télécharger les ressources nécessaires dans ce dossier
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Chargement du fichier texte (change le chemin selon ton fichier !)
with open("pride_and_prejudice.txt", "r", encoding='utf-8') as file:
    data = file.read().replace('\n', ' ')

# Tokenisation des phrases
sentences = sent_tokenize(data)

# Prétraitement de chaque phrase
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in string.punctuation]
    return words

# Prétraitement du corpus complet
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# Fonction de similarité basée sur l’intersection Jaccard
def get_most_relevant_sentence(query):
    query_tokens = preprocess(query)
    max_similarity = 0
    best_sentence = "Sorry, I don't understand your question."

    for i, sentence_tokens in enumerate(preprocessed_sentences):
        union = set(query_tokens).union(set(sentence_tokens))
        intersection = set(query_tokens).intersection(set(sentence_tokens))
        if union:
            similarity = len(intersection) / len(union)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentences[i]  # Retourne la phrase originale (non prétraitée)

    return best_sentence

# Fonction principale Streamlit
def main():
    st.title("📚 Chatbot - Pride and Prejudice")
    st.write("Hello! I'm your chatbot trained on *Pride and Prejudice*. Ask me anything!")

    user_input = st.text_input("You: ")

    if st.button("Submit") and user_input.strip() != "":
        response = get_most_relevant_sentence(user_input)
        st.markdown(f"**Chatbot:** {response}")

# Exécution
if __name__ == "__main__":
    main()
