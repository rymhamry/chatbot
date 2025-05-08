import os
import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 📁 Configuration du dossier local pour les ressources NLTK
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)  # Ajouter en tête

# ⬇️ Téléchargement des ressources NLTK si non présentes
nltk_resources = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet"
}

for resource, path in nltk_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# 📖 Chargement du fichier texte (assure-toi que ce fichier est bien présent dans ton repo)
with open("pride_and_prejudice.txt", "r", encoding="utf-8") as file:
    data = file.read().replace("\n", " ")

# 🔤 Tokenisation des phrases
sentences = sent_tokenize(data)

# 🧹 Prétraitement
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(sentence):
    words = word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in string.punctuation]

preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# 🔍 Fonction de similarité Jaccard
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
                best_sentence = sentences[i]

    return best_sentence

# 🚀 App Streamlit
def main():
    st.title("📚 Chatbot - Pride and Prejudice")
    st.write("Hello! I'm your chatbot trained on *Pride and Prejudice*. Ask me anything!")

    user_input = st.text_input("You: ")

    if st.button("Submit") and user_input.strip():
        response = get_most_relevant_sentence(user_input)
        st.markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()