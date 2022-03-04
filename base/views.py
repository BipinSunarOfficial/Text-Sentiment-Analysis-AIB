from django.shortcuts import render , HttpResponse
import numpy as np
import pandas as pd
import re #for regular expressions
import nltk #for text manipulation
import string
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


def home(request):
    return render(request,('base/home.html'))


def nb(request):
    if request.method == 'POST':
        text = request.POST['search']
        pd.set_option("display.max_colwidth",200)
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        train =pd.read_csv(r'C:\Users\Bipin\Desktop\Sentiment analysis\TSA\data\train\train.csv', encoding='utf-8')
        train
        test = pd.read_csv(r'C:\Users\Bipin\Desktop\Sentiment analysis\TSA\data\test\test.csv', encoding='utf-8')
        test
        train[train['label'] == 0].head(10)
        train[train['label'] == 1].head(10)
        train.shape
        test.shape
        train['label'].value_counts()
        length_train_dataset = train['tweet'].str.len()
        length_test_dataset = test['tweet'].str.len()
        # plt.hist(length_train_dataset, bins=20,label="Train tweets")
        # plt.hist(length_test_dataset, bins=20,label="Test tweets")
        # plt.legend() 
        # plt.show()
        combine=train.append(test,ignore_index=True) #train and test dataset are combined
        combine.shape
def remove_pattern(input_text,pattern):
        r= re.findall(pattern, input_text)
        for i in r:
        input_text = re.sub(i, '', input_text)
        return input_text

# Create your views here.
