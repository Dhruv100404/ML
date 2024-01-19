import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
import pickle
from keras.preprocessing.sequence import pad_sequences


# Download NLTK resources
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

current_directory = os.path.abspath(os.path.dirname(__file__))
model_weights_path = os.path.join(current_directory, 'model1.pkl')
tokenizer_path = os.path.join(current_directory, 'tokenizer.pkl')

model = pickle.load(open(model_weights_path, 'rb'))
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

# Streamlit App
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess and predict
    transformed_sms = transform_text(input_sms)
    vector_input = tokenizer.texts_to_sequences([transformed_sms])
    vector_input = pad_sequences(vector_input, maxlen=500, padding='post')
    result = model.predict(vector_input)[0]

    # Display result
    if result > 0.5:
        st.header("Spam")
    else:
        st.header("Not Spam")
