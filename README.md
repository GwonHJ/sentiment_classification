# NLP

## 졸업프로젝트 : 한국어 문장을 감정 3가지로 분류하기

classification.py
mk_dataset.py
requirements.txt

 - 데이터셋 : 한국어 감정 정보가 포함된 단발성 데이터셋, 한국어 감정 정보가 포함된 연속적 데이터셋([AI Hub의 오픈데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence)), 네이버 쇼핑리뷰
 - mk_dataset.py : 네이버 쇼핑리뷰를 감정분류에 편하게 이용하도록 .txt -> .csv로 변경하는 코드
 - classification.py : Lstm을 이용하여 감정을 긍정, 중립, 분류 3가지로 분류하는 코드


### 필요한 과정 : 감정정보가 포함된 데이터셋을 3가지로 변경, 환경 맞추기
 
 #### 기준
  - 긍정(1) : 행복
  - 중립(0) : 중립
  - 부정(-1) : 분노, 혐오, 슬픔, 공포
  놀람은 긍정, 부정, 중립으로 나누기 애매하여 데이터를 제거하였습니다.

 #### requirements.txt를 통해서 환경 맞춰주기



## [Classifying korean sentence](https://github.com/GwonHJ/NLP/tree/master/Classifying%20korean%20sentence) : 한국어 문장 감정 분석


꿈꾸는 아이 대회 https://dacon.io/competitions/official/235664/codeshare/ 출전을 위해서 자연어처리 챗봇 개발 프로젝트

맡은 역할 : KoLPy, Keras, LSTM등을 이용하여 한국어 문장에 대한 감정을 분류, 모델 사용을 위한 인터페이스(챗봇ui)

#### 데이터 출처

  - train 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 단발성 데이터셋

  - test 데이터 : https://aihub.or.kr/keti_data_board/language_intelligence 에서 한국어 감정 정보가 포함된 연속적 데이터셋 
    

데이터가 7가지 감정으로 분류되어있으나 데이터의 분류가 정확하지 않은 관계로 긍정, 부정 두가지 분류로만 훈련시켰습니다.
 
데이터 전처리 및 분류
https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
참고하여 만들었습니다.

#### 개발 전 해야하는 일 
  1. AIHUB에 가입하여 오픈데이터셋을 받는다.
  2. 데이터셋을 긍정, 부정 두가지로 분류한다 -> 공포, 혐오, 분노, 슬픔 : 0(부정), 행복 : 1(긍정)
  

링크에서는 데이터셋이 txt 파일이기 때문에 그 전 과정이 조금 다름.
- [train.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/train.py) : train데이터셋을 형태소 분석하여 json파일로 만들기
- [test.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/test.py) : test데이터셋을 형태소 분석하여 json파일로 만들기
- [after_json.py](https://github.com/GwonHJ/NLP/blob/master/Classifying%20korean%20sentence/after_json.py) : json파일을 가지고 전처리 및 훈련


#### <위에 참고한 링크와 다른 부분 설명 : cvs파일을 json파일로 만드는 과정>

```python

data=pd.read_csv('onlybinary_data.csv', sep=",")

#print(data.head()) 출력확인
##print(len(data)) 크기확인
##print(type(data)) 타입확인


train_data = pd.DataFrame(data, columns=['Sentence','Emotion'])

#json에 넣기 위해서 after_data에 담아줌
after_data = []

for index in range(len(train_data)):
    okt = Okt()
    ##실행이 잘 되고 있는지 확인용
    if index%50 ==0:
        print(index)
        
    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)] 

    tokens= tokenize(train_data['Sentence'][index])    
    feeling=train_data['Emotion'][index]
    feeling=int(feeling)
    ##json파일 형태가 [[단어/형태소, 단어/형태소, ..., 단어/형태],[감정]] 이런 형태가 되어야 하기 때문에 [[tokens, feeling]] 
    after_data = after_data + [[tokens,feeling]]

```

### 형태소 분석후 빈수가 높은 단어 50개를 그래프로 나타낸것(Train_data에서)

![형태소분석후빈도수상위50개](https://user-images.githubusercontent.com/45057466/98559748-59402600-22ea-11eb-8e7c-c7a630b4b1c5.png)


### 결과
![KakaoTalk_20201109_200511936](https://user-images.githubusercontent.com/45057466/98559780-6230f780-22ea-11eb-9bf0-77d1834f79df.png)


### 아쉬운점 
  1. dataset의 감정분류가 어떻게 분류된것인지 몰라서 적절하지 않은 경우도 있다. 
  2. 데이터셋이 부족하다.
  3. 데이터셋이 무작위로 크롤링한 데이터 같아서 경연대회에 참가할 챗봇용 데이터에는 적합하지 않아보인다....


### [chatbotUI](https://github.com/GwonHJ/NLP/tree/master/chatboyUI) : 챗봇UI


chatbot.ui : 레이아웃 파일 

chatbot.py : 이벤트 다 연결해놓은 코드


PyQt를 이용하여 만든 챗봇 UI

Qt Designer를 이용해서 레이아웃을 만들었습니다.(.ui파일)

#### 실행화면

![KakaoTalk_20201112_160735705](https://user-images.githubusercontent.com/45057466/98907315-fa54f980-2501-11eb-9d0a-6c5313c5b3a3.png)

아직 챗봇모델과 연결하지 않은 상태라서 크누아이가 안녕만 답하고 있습니다. 챗봇모델과 연결하면 답변이 제대로 작동함.

https://wikidocs.net/book/2944
를 참고하여 만들었습니다.

-------------------------------------------------------------------------------------


## [Ex](https://github.com/GwonHJ/NLP/tree/master/Ex) : 위키독스에 있는 '딥 러닝을 이용한 자연어 처리 입문' 실습예제를 따라한 코드

https://wikidocs.net/book/2155
