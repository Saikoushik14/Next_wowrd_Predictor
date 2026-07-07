# 📖 Next Word Prediction using Simple RNN (Many-to-Many RNN)

## Overview

This project demonstrates a **Many-to-Many Recurrent Neural Network (RNN)** using **TensorFlow/Keras** to perform **next word prediction**. The model is trained on the **Sherlock Holmes** text dataset and predicts the most probable next word for a given input sentence. A user-friendly **Streamlit** interface is provided to interact with the trained model.

---

## Features

* Many-to-Many RNN implementation using **SimpleRNN**
* Text preprocessing and cleaning
* Tokenization and word indexing
* Sliding window sequence generation
* Word Embedding layer
* Next-word prediction with confidence score
* Interactive Streamlit web application
* Model and tokenizer saved for future predictions

---

## Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy
* Pandas
* Streamlit
* Pickle

---

## Project Structure

```text
Many_to_Many_RNN/
│
├── app.py                  # Main application
├── sherlock.txt            # Training dataset
├── next_word_model.keras   # Saved trained model
├── tokenizer.pkl           # Saved tokenizer
├── requirements.txt
└── README.md
```

---

## Dataset

The model is trained on the **Sherlock Holmes** text corpus.

Dataset contains:

* English text paragraphs
* Thousands of words
* Used for learning word sequences

---

## Model Architecture

```text
Input Text
      │
      ▼
Text Cleaning
      │
      ▼
Tokenizer
      │
      ▼
Sliding Window Sequences
      │
      ▼
Embedding Layer
      │
      ▼
SimpleRNN Layer
      │
      ▼
Dense Layer (ReLU)
      │
      ▼
Output Layer (Softmax)
      │
      ▼
Predicted Next Word
```

---

## Installation

Clone the repository:

```bash
git clone <repository_link>
```

Move to the project directory:

```bash
cd Many_to_Many_RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Example Predictions

| Input           | Predicted Word |
| --------------- | -------------- |
| my dear         | watson         |
| i have          | been           |
| there was       | a              |
| he was          | less           |
| sherlock holmes | i              |

---

## Workflow

1. Load Sherlock Holmes dataset.
2. Clean and preprocess the text.
3. Tokenize the words.
4. Generate input-output sequences using a sliding window.
5. Train the SimpleRNN model.
6. Save the trained model and tokenizer.
7. Accept user input through Streamlit.
8. Predict the next word along with its confidence score.

---

## Applications

* Text Autocompletion
* Smart Keyboard Suggestions
* AI Writing Assistants
* Chatbots
* Natural Language Processing
* Language Modeling

---

## Future Enhancements

* Replace **SimpleRNN** with **LSTM** or **GRU** for better performance.
* Display the **Top 5 predicted words** instead of only one.
* Train on a larger text corpus.
* Improve the Streamlit UI.
* Add model accuracy and loss graphs.

---

## Learning Outcomes

* Understanding of Many-to-Many RNN architecture
* Text preprocessing techniques
* Tokenization and sequence generation
* Word embedding concepts
* Training neural networks for NLP tasks
* Building interactive ML applications using Streamlit

---

## Author

**SAI KOUSHIK KASULA**

**B.Tech – Computer Science and Engineering (Data Science)**

**Anurag University**

---

## License

This project is developed for **educational and academic purposes**.
