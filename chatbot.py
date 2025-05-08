import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import streamlit as st


with open("C:/Users/HP/Pictures/Camera Roll/doc formation/streamlit/.venv/chatbot/pride_and_prejudice.txt", "r", encoding='utf-8') as file:
    data = file.read().replace('\n', ' ')



sentences = sent_tokenize(data)

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


vocab = [preprocess(sentence) for sentence in sentences]


def similarity(query):
    query = preprocess(query)

    max_similarity = 0
    for sentence in vocab:
        similarity = len(set(query).intersection(sentence)) / len(set(query).union(sentence))
        if similarity > max_similarity:
            max_similarity = similarity
            relevant_sentence = ' '.join(sentence)
    return relevant_sentence


def main():
    st.title('ChatBot')
    st.write("Hello! I'm your chatbot. Ask me anything")

    query = st.text_input('You: ')
    if st.button('Submit'):
        response = similarity(query)
        st.write('Chatbot : ' + response)



if __name__ == "__main__":
    main()