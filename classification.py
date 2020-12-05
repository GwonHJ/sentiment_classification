# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:02:51 2020

@author: 현
"""
import pandas as pd
import matplotlib.pyplot as plt
import konlpy
from konlpy.tag import Okt 
from keras.preprocessing.text import Tokenizer 
import numpy as np



#수정_합본데이터 : 단발성+연속성 데이터 불러오기
train_data = pd.read_csv("수정_합본.csv")
#네이버리뷰 추가
train_data2 = pd.read_csv("sample.csv")
train_data2 = train_data2[:80000]
train_data = pd.concat([train_data,train_data2])

print(train_data.groupby('Emotion').size().reset_index(name='count')) 

#%%sentence전처리
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

print(1)

okt = Okt()
#불용어 제거
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

X_train = []
#돌아가는지 확인용...
cnt=-1 
for sentence in train_data['Sentence']: 
    cnt = cnt +1 
    if cnt%2000==0:
        print(cnt)
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X) 


max_words = 35000
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 

#%%


print(2)
print("문장의 최대 길이 : ", max(len(l) for l in X_train)) 
print("문장의 평균 길이 : ", sum(map(len, X_train))/ len(X_train)) 
'''
그래프
plt.hist([len(s) for s in X_train], bins=50) 
plt.xlabel('length of Data') 
plt.ylabel('number of Data') 
plt.show()
'''
#%%

print(3)
y_train = []

for i in range(len(train_data['Emotion'])): 
    if train_data['Emotion'].iloc[i] == 1: 
        y_train.append([0, 0, 1]) 
    elif train_data['Emotion'].iloc[i] == 0:
        y_train.append([0, 1, 0]) 
    elif train_data['Emotion'].iloc[i] == -1:
        y_train.append([1, 0, 0])

y_train = np.array(y_train)



#%%데이터셋 나누기
from sklearn.model_selection import train_test_split

print(4)
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 100)


#%%

from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.models import Sequential 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences 
from keras.layers import BatchNormalization
import keras

print(5)
max_len = 20 # 전체 데이터의 길이를 20로 맞춘다 

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''

#%%
print(6)
model = Sequential()
model.add(Embedding(max_words,128))
model.add(LSTM(64, return_sequences = True))
model.add(BatchNormalization())
model.add(Dropout(0.6)) # 드롭아웃 추가. 비율은 50%
model.add(LSTM(32, return_sequences = False))
model.add(Dropout(0.4)) # 드롭아웃 추가. 비율은 50%
model.add(BatchNormalization())
model.add(Dense(16, activation='relu')) 
model.add(Dense(9, activation='relu')) 
model.add(Dense(3, activation='softmax')) 
  
'''
##과적합이 일어날거같으면 중지
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
'''
mc = ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

opt = keras.optimizers.Adam(lr=0.00002)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc']) 
history = model.fit(x_train, y_train, batch_size=100, epochs=30, callbacks=[mc], validation_data=(x_test, y_test))
##
#로스, 정확도 변화 그래프
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

epochs = range(1, len(history.history['acc']) + 1) 
plt.plot(epochs, history.history['acc'])  
plt.plot(epochs, history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% 입력된 문장 예측값

from tensorflow.keras.models import load_model

loaded_model = load_model('best_model2.h5')
print("ddd")


def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  print(pad_new)
  score = loaded_model.predict(pad_new) # 예측
  print(score)
  max = np.argmax(score)
  
  if max == 0:
    print("부정")
  elif max == 1:
    print("중립")
  else :
    print("긍정")
  
  
temp_str = ""

while temp_str != 'stop':
    temp_str = input()
    sentiment_predict(temp_str)
