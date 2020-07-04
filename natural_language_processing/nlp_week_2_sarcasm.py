import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt

# Specify hyper-parameters
vocab_size = 10000
embedding_size = 16
max_length = 32
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 20000

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

# Split into training and validation sets
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Create a Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# Create a word index
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)
# Use same tokenizer in test data - a lot more OOVs might be created
testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length, truncating=trunc_type)

# Define a Sequential Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

# Use binary cross-entropy as we have two classes
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

