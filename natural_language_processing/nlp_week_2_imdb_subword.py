import tensorflow as tf
import tensorflow_datasets as tfds


# Import IMDB dataset that has already been tokenised
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
# Split test/training sets
train_data, test_data = imdb["train"], imdb["test"]

# Reshape data
# train_data = train_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
# test_data = test_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

# Access tokeniser
tokenizer = info.features["text"].encoder
print(tokenizer.subwords)

# Example of how it encodes/decodes string
sample_string = "TensorFlow, from basics to mastery"
tokenized_string = tokenizer.encode(sample_string)
original_string = tokenizer.decode(tokenized_string)

for ts in tokenized_string:
    print(f"{ts} ---> {tokenizer.decode([ts])}")

# Now let's use for classification
embedding_dim = 64
vocab_size = tokenizer.vocab_size

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),  # Shape of vectors from tokenizer are not easily flattened otherwise
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
    tf.keras.layers.Reshape(())])

model.summary()

# Compile and train model
num_epochs = 10

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(train_data, epochs=num_epochs, validation_data=test_data)
