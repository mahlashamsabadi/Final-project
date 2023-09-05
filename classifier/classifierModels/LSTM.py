from django.shortcuts import render
from transformers import TFBertForSequenceClassification
from transformers import BertConfig, BertTokenizer
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features
import pandas as pd 
import tensorflow as tf
from django.http import HttpResponse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import AutoConfig, AutoTokenizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import hazm
import re
from collections import Counter
from classifier.dataPrepration.preprocess import cleaning

    # Count unique words
def counter_word1(text):
    count = Counter()
    for word in text.split():
      count[word] += 1
    return count
def classifyWithLSTM(test_string):
    normalizer = hazm.Normalizer()
    stopwords = hazm.stopwords_list
    tokenizer = hazm.word_tokenize
    new_model = keras.models.load_model("D:\\term8\\final_project\\LSTM-model")
    # test_string = 'سازمان جهانی بهداشت: چین جهش بزرگی در شمار بیماران بستری مبتلا به کرونا را گزارش کرد '
    test_string = cleaning(test_string , normalizer,stopwords,tokenizer)
    counter1 = counter_word1(test_string)
    num_unique_words1 = len(counter1)
    num_unique_words1
    test_string = [test_string]
    # vectorize a text corpus by turning each text into a sequence of integers
    tokenizer1 = Tokenizer(num_words=num_unique_words1)
    tokenizer1.fit_on_texts(np.asarray(list(test_string)))

    word_index1 = tokenizer1.word_index
    train_sequences1 = tokenizer1.texts_to_sequences(test_string)
    # Pad the sequences to have the same length

    # Max number of words in a sequence
    max_length1 = 20

    train_padded1 = pad_sequences(train_sequences1, maxlen=max_length1, padding="post", truncating="post")
    
    predictions = new_model.predict(train_padded1)
    print(predictions)
    predictions = [1 if p > 0.1 else 0 for p in predictions]
    return predictions