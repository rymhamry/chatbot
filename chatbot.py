import os
import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 📁 Configurer le dossier NLTK local
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# ⬇️ Télécharger les ressources nécessaires
def download_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }
    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, download_dir=nltk_data_dir)

# 🧹 Fonction de prétraitement
def preprocess(sentence, lemmatizer, stop_words):
    words = word_tokenize(sentence.lower())
    return [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word not in string.punctuation
    ]

# 🔍 Recherche de la phrase la plus pertinente
def get_most_relevant_sentence(query, sentences, preprocessed_sentences, lemmatizer, stop_words):
    query_tokens = preprocess(query, lemmatizer, stop_words)
    max_similarity = 0
    best_sentence = "Sorry, I don't understand your question."

    for i, sentence_tokens in enumerate(preprocessed_sentences):
        union = set(query_tokens).union(set(sentence_tokens))
        intersection = set(query_tokens).intersection(set(sentence_tokens))
        if union:
            similarity = len(intersection) / len(union)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentences[i]
    return best_sentence

# 🚀 Lancement de l'app
def main():
    st.title("📚 Chatbot - Pride and Prejudice")
    st.write("Hello! I'm your chatbot trained on *Pride and Prejudice*. Ask me anything!")

    # 📥 Télécharger les ressources NLTK
    download_nltk_resources()

    # 🔧 Initialiser lemmatiseur et stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # 📄 Charger le texte du fichier
    with open("pride_and_prejudice.txt", "r", encoding="utf-8") as file:
        data = file.read().replace("\n", " ")

    # ✂️ Tokenisation après que punkt est dispo
    sentences = sent_tokenize(data)
    preprocessed_sentences = [preprocess(s, lemmatizer, stop_words) for s in sentences]

    # 💬 Interaction utilisateur
    user_input = st.text_input("You:")
    if st.button("Submit") and user_input.strip():
        response = get_most_relevant_sentence(user_input, sentences, preprocessed_sentences, lemmatizer, stop_words)
        st.markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()
