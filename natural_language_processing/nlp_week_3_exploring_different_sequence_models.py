import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt

# Specify hyper-parameters
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 20000

# Load data
cdw = os.getcwd()
path = os.path.join(cdw, "natural_language_processing/Sarcasm_Headlines_Dataset.json")

dataStore = []
with open(path) as inputData:
    for line in inputData:
        try:
            dataStore.append(json.loads(line.rstrip(';\n')))
        except ValueError:
            print("Skipping invalid line {0}".format(repr(line)))

sentences, labels = [], []

for item in dataStore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])

# Split into training/test set
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Create a Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# Get word indices
word_index = {value: key for key, value in tokenizer.word_index.items()}

# Tokenize sentences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)
# Use same tokenizer in test data - a lot more OOVs might be created
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

# Convert to numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

# or, model with 1D convolutional layer
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     tf.keras.layers.Conv1D(128, 5, activation='relu'),
#     tf.keras.layers.GlobalMaxPooling1D(),
#     tf.keras.layers.Dense(24, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')])


model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)


# PLot results
def plot_graphs(hist, string):
    plt.plot(hist.history[string])
    plt.plot(hist.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
