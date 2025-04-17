import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model("lstm_textgen_modell.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")



file_path = tf.keras.utils.get_file(
    'shakespeare.txt',
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

characters = sorted(set(text))
char2ind = {ch: idx for idx, ch in enumerate(characters)}
ind2char = {idx: ch for ch, idx in char2ind.items()}

seq_len = 40


def one_hot_encode_input(input_text, seq_len):
    x_pred = np.zeros((1, seq_len, len(characters)))
    for t, char in enumerate(input_text.rjust(seq_len)[-seq_len:]):
        if char in char2ind:
            x_pred[0, t, char2ind[char]] = 1
    return x_pred


def predict_next_char(seed_text):
    x_pred = one_hot_encode_input(seed_text, seq_len)
    prediction = model.predict(x_pred)[0]
    next_index = np.argmax(prediction)
    return ind2char[next_index]


def generate_text(seed, length):
    result = seed
    for _ in range(length):
        next_char = predict_next_char(result)
        result += next_char
    return result

st.title("LSTM Shakespeare Text Generator")

seed = st.text_input("Enter your seed text:", "To be, or not to be")
length = st.slider("Characters to generate:", 50, 1000, 300)

if st.button("Generate"):
    with st.spinner('Generating text...'):
        generated = generate_text(seed, length)
    st.subheader("Generated Text")
    st.write(generated)
