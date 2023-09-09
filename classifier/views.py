from django.shortcuts import render , redirect
from django.http import HttpResponse
from .forms import ClassifyForm
from classifier.classifierModels.ParseBert import classifyWithParseBert
from classifier.classifierModels.LSTM import classifyWithLSTM 
from classifier.classifierModels.GRU import classifyWithGRU 
from classifier.classifierModels.CNN import classifyWithCNN 
from classifier.classifierModels.Albert import classifyWithAlbert 
from django.contrib import messages
from classifier.classifierModels.XlmRoberta import classifyWithXlmr 
# Create your views here.
def classifyNews(request):
    
    if request.method == 'POST':
        form = ClassifyForm(request.POST)
        print('1')
        result = 'nothing'
        if form.is_valid():
            print('2')
            data = form.cleaned_data
            print(data['text'])
            print('2')
            if data['model'] == 'ParseBert':
                print('true')
                result = classifyWithParseBert(data['text'])
            elif data['model'] == 'Albert':
                result = classifyWithAlbert(data['text'])
            elif data['model'] == 'LSTM':
                result = classifyWithLSTM(data['text'])
            elif data['model'] == 'CNN':
                result = classifyWithCNN(data['text'])
            elif data['model'] == 'GRU':
                result = classifyWithGRU(data['text'])
            else :
                result = classifyWithXlmr(data['text'])
        print(result)
        
        if result[0] == 0:
            result = 'fact'
        else:
            result = 'fake'
        print(result)
        messages.add_message(request, messages.INFO, f"This news is '{result}'")
        return render(request , 'page.html',{'form':form})
    else:
        form = ClassifyForm()
        return render(request , 'page.html',{'form':form})
