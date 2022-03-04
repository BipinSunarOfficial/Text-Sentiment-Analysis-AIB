import numpy as np
import pandas as pd
train = pd.read_csv(r'C:\Users\Bipin\Desktop\Sentiment analysis\TSA\data\test\test.csv', encoding='utf-8')
train.head()
test = pd.read_csv(r'C:\Users\Bipin\Desktop\Sentiment analysis\TSA\data\train\train.csv', encoding='utf-8')
test.head()
train[train['label'] == 0].head(10)