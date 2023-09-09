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
import hazm
import re

def classifyWithAlbert(text):
    # loaded_model = TFBertForSequenceClassification.from_pretrained('D:\\term8\\final_project\\AlbertModel')
    # # lines = list(text)
    df1 = pd.DataFrame(columns=['text', 'label'])
    df1.loc[0] = [text, "fact"]
    # print(text)
    id_label_map = {
        'fact': 0,
        'fake': 1
    }
    df1['label'] = df1['label'].map(id_label_map)

    MODEL_NAME = "m3hrdadfi/albert-fa-base-v2-clf-digimag"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    x_test1 = tokenizer(df1['text'].tolist(), padding='max_length', truncation=True, max_length=MAX_LEN)
    y_test1 = df1['label'].values
    x_test1 = {key: np.array(val) for key, val in x_test1.items()}
    test_dataset1 = tf.data.Dataset.from_tensor_slices((x_test1, y_test1)).batch(VALID_BATCH_SIZE)

    # optimizer1 = tf.keras.optimizers.Adam(learning_rate=3e-5)
    # loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loaded_model.compile(optimizer1, loss1)

    # ev = loaded_model.evaluate(test_dataset1)
    # print()
    # print(f'Evaluation: {ev}')
    # print()

    # predictions = loaded_model.predict(x_test1)
    # ypred1 = predictions[0].argmax(axis=-1).tolist()
    # return ypred1
    LEARNING_RATE = 2e-5
    MODEL_NAME = "m3hrdadfi/albert-fa-base-v2-clf-digimag"
    model2 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model2.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model2.load_weights('D:\\term8\\final_project\\Albert\\weights')
    predictions = model2.predict(x_test1)
    ypred1 = predictions[0].argmax(axis=-1).tolist()
    return ypred1