from django import forms

class ClassifyForm(forms.Form):
    models = (
        ('ParseBert','ParseBert'),
        ('Albert','Albert'),
        ('LSTM','LSTM'),
        ('GRU','GRU'),
        ('CNN','CNN'),
        ('xlm-roberta','xlm-roberta'),
    )
    text = forms.CharField()
    model = forms.ChoiceField(choices = models)