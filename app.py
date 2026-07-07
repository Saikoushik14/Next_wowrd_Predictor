# ==============================================
# Many-to-Many RNN
# Next Word Prediction using Sherlock Holmes
# ==============================================

import os
import re
import pickle
from tracemalloc import start
from matplotlib import lines
import numpy as np
import streamlit as st

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==============================================
# Configuration
# ==============================================

DATASET = "1661-0.txt"
MODEL = "next_word_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000

# ==============================================
# Clean Text
# ==============================================

def clean_text(text):
    text = text.lower()

    text = re.sub(r"\r", " ", text)
    text = re.sub(r"\n", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================
# Train Model (Sliding Window)
# ==========================================

def train_model():

    print("Loading Dataset...")

    with open(DATASET, "r", encoding="utf-8") as file:
        text = file.read()

    # Remove Project Gutenberg header
    start = text.find("I. A SCANDAL IN BOHEMIA")
    if start != -1:
        text = text[start:]

    # Keep only first 60000 words
    words = text.split()[:60000]
    text = " ".join(words)

    text = clean_text(text)

    print("Dataset Loaded")
    print("Total Words :", len(text.split()))

    tokenizer = Tokenizer(
        num_words=5000,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts([text])

    total_words = min(5000, len(tokenizer.word_index) + 1)

    print("Vocabulary :", total_words)

    sequence_length = 10

    input_sequences = []

    tokens = tokenizer.texts_to_sequences([text])[0]

    for i in range(sequence_length, len(tokens)):

        seq = tokens[i-sequence_length:i+1]

        input_sequences.append(seq)

    input_sequences = np.array(input_sequences)

    X = input_sequences[:, :-1]

    y = input_sequences[:, -1]

    y = to_categorical(y, num_classes=total_words)

    print("Input Shape :", X.shape)
    print("Output Shape :", y.shape)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(
            (tokenizer, sequence_length),
            f
        )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=total_words,
            output_dim=128,
            input_length=sequence_length
        )
    )

    model.add(
        SimpleRNN(
            256
        )
    )

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        Dense(
            total_words,
            activation="softmax"
        )
    )

    model.summary()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X,
        y,
        epochs=50,
        batch_size=128,
        verbose=1
    )

    model.save(MODEL)

    print("Model Saved")
# ==========================================
# Predict Next Word
# ==========================================

def predict_next_word(seed_text):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer, sequence_length = pickle.load(f)

    seed_text = clean_text(seed_text)

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Keep only last 5 words
    token_list = token_list[-sequence_length:]

    token_list = pad_sequences(
        [token_list],
        maxlen=sequence_length,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            predicted_word = word
            break

    confidence = prediction[0][predicted_index]

    return predicted_word, confidence

# ==========================================
# Train Model (Only Once)
# ==========================================

if not os.path.exists(MODEL):
    train_model()

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Next Word Prediction using Simple RNN")

st.markdown("### Many-to-Many RNN")

st.write("Enter a sentence and the model will predict the next word.")

sentence = st.text_input(
    "Enter Text",
    placeholder="Example : sherlock holmes"
)

if st.button("Predict"):

    if sentence.strip() == "":
        st.warning("Please enter some text.")

    else:

        word, confidence = predict_next_word(sentence)

        st.success(f"Predicted Word : {word}")

        st.info(f"Confidence : {confidence*100:.2f}%")