import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
