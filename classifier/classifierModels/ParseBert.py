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
from classifier.dataPrepration.makeExample import make_examples
def classifyWithParseBert(text):
    loaded_model = TFBertForSequenceClassification.from_pretrained('D:\\term8\\final_project\\ParsBertModel')
    print('********************************************************')
    MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    print('********************************************************')
    # with open('./1.txt') as f:
    #     lines = f.readlines()
    # print(lines)
    # # lines = request.POST.get('user_input', '')
    # print(lines)
    # lines = list(text)
    df1 = pd.DataFrame(columns=['text', 'label'])
    df1.loc[0] = [text, "fact"]
    print(df1)
    id_label_map = {
    'fact': 0,
    'fake': 1
    }
    df1['label'] = df1['label'].map(id_label_map)
    x_test1, y_test1 = df1['text'].values.tolist(), df1['label'].values.tolist()
    print('********************************************************')
    test_dataset_base1, test_examples1 = make_examples(tokenizer, x_test1, y_test1, maxlen=128)
    [xtest1, ytest1], test_examples1 = make_examples(tokenizer, x_test1, y_test1, maxlen=128, is_tf_dataset=False)
    print('********************************************************')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loaded_model.compile(optimizer, loss)
    # ev = loaded_model.evaluate(test_dataset_base1.batch(8))
    # print()
    # print(f'Evaluation: {ev}')
    # print()
    
    predictions = loaded_model.predict(xtest1)
    ypred1 = predictions[0].argmax(axis=-1).tolist()
    return ypred1





