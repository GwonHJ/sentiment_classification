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
train_data2 = train_data2[:100000]
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
    

max_words = 38000
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
#원핫인코딩
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
max_len = 15 # 전체 데이터의 길이를 15로 맞춘다 

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
model.add(Dropout(0.6)) # 드롭아웃 추가. 비율은 60%
model.add(LSTM(32, return_sequences = False))
model.add(BatchNormalization()) 
model.add(Dropout(0.2)) # 드롭아웃 추가. 비율은 20%
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1)) # 드롭아웃 추가. 비율은 20%
model.add(Dense(9, activation='relu')) 
model.add(Dense(3, activation='softmax'))

print(7)  

mc = ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

opt = keras.optimizers.rmsprop(lr=0.00003)
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
    print("부정 : ",end="")
  elif max == 1:
    print("중립 : ",end="")
  else :
    print("긍정 : ",end="")
  return max

    
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=W0613, C0116
# type: ignore[union-attr]
# This program is dedicated to the public domain under the CC0 license.
import random
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

def help(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('사용자 채팅의 감정을 판단하여\n'
                              '긍정,중립,부정을 3가지로 분류하여\n'
                              '이모티콘을 답장해줌')

def reply(update: Update, context: CallbackContext) -> None:
    # text : 사용자가 입력한 채팅, max : 감정의 index
    text = update.message.text
    max = sentiment_predict(text)
    print(text, end="")
    print(max)
    
    # 이모티콘 리스트(부정,중립,긍정 순)
    emo = [["😟","😕","😡","😨","😱"],["😐","🙄"],["😀","😃","😄","😁","😆","😊","🙂","😍","😋","🤗"]]
    
    # 버튼 생성
    show_list = []
    show_list.append(InlineKeyboardButton("Good",callback_data="Good"))
    show_list.append(InlineKeyboardButton("Bad",callback_data="Bad"))
    show_markup = InlineKeyboardMarkup(build_box(show_list,2))
    
    # 답장할 이모티콘
    emotion = emo[max][int(random.random()*len(emo[max]))]
    update.message.reply_text(emotion)
    update.message.reply_text("정확한가요?",reply_markup=show_markup)

def error(bot, update, error):
    logger.warning("Update %s caused error %s",update,error)

def build_box(buttons, n_cols, header_buttons=None, footer_buttons=None):
    menu = [buttons[i:i+n_cols] for i in range(0,len(buttons),n_cols)]
    if header_buttons:
        menu.insert((0,header_buttons))
    if footer_buttons:
        menu.append(footer_buttons)
    return menu

def callback_get(update: Update, context: CallbackContext):
    print("Callback")
    if update.callback_query.data == "Good":
        update.callback_query.message.reply_text("Good")
        return 0;
    elif update.callback_query.data == "Bad":
        update.callback_query.message.reply_text("BAD")
        return 1;
    else:
        update.callback_query.message.reply_text("아무것도 아니다")

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    my_token = '1405029887:AAH0tk74Vhv01H6iiHsYTBKAi3UXRmrEH7o'
    updater = Updater(my_token, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("help", help))
    # on noncommand i.e message
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, reply))
    dispatcher.add_handler(CallbackQueryHandler(callback_get))
    # error handler
    dispatcher.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
