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




def cleaning(text , normalizer,stopwords,tokenizer):
    text = text.strip()
    # normalizing
    text = normalizer.normalize(text)
    # removing wierd patterns
    tokens = tokenizer(text)
    filtered = list(set(tokens) - set(stopwords()))
    # Define a pattern to match punctuation characters (including Arabic and Persian punctuation)
    punctuation_pattern = list('،.؛؟!"#$%&\'()*+/:;<=>?@[\\]^_`{|}~')
    # Remove punctuation from the tokens
    filtered = list(set(filtered) - set(punctuation_pattern))
    # Join the filtered tokens back into a single string
    text = ' '.join(filtered)
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)

    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)

    return text
