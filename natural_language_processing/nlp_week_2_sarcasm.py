import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Specify hyper-parameters
vocab_size = 10000
embedding_size = 16
max_length = 120
trunc_type = "post"
oov_tok = "<OOV>"

