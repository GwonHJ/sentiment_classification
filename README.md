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
