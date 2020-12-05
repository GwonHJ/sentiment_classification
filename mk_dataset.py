# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:56:43 2020

@author: 현
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



total_data = pd.read_table('naver_shopping.txt', names=['ratings', 'Sentence'])
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력



total_data[:5]


total_data['Emotion'] = np.select([total_data.ratings > 3], [1], default=-1)
total_data[:5]


total_data['ratings'].nunique(), total_data['Sentence'].nunique(), total_data['Emotion'].nunique()



total_data.drop_duplicates(subset=['Sentence'], inplace=True) # reviews 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(total_data))



print(total_data.isnull().values.any())

#%%
df = total_data.drop(['ratings'], axis = 'columns', inplace = True)

total_data.to_csv('sample.csv', encoding = 'utf-8')

#%%



data = pd.read_table('sample.csv', encoding = 'utf-8')

data = data[:100000]

