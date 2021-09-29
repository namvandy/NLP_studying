# 자연어 처리 전처리 이해하기
# 토큰화, 단어 집합 생성, 정수 인코딩, 패딩, 벡터화의 과정
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# 토큰화(Tokenization)
# 주어진 텍스트를 단어 또는 문자 단위로 자르는 것
# 영어의 경우 spaCY, NLTK
en_text = "A Dog RUn back corner near spare bedrooms"
# spaCY 사용
import spacy
spacy_en = spacy.load('en_core_web_sm')
def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]
print("spaCY사용: ",tokenize(en_text))

# NLTK 사용
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
print("NLTK사용: ",word_tokenize(en_text))

# 띄어쓰기로 토큰화
print("띄어쓰기로: ", en_text.split())

# 단어집합(vocabulary)란 중복제거한 텍스트의 총 단어의 집합(set)

# 한국어 띄어쓰기 토큰화
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print("한국어띄어쓰기로: ",kor_text.split())
# 사과란 단어가 4번 등장하는데 모두 조사가 붙어있기 때문에 제거하지 않으면 기계는 같은 단어로 인식

# 형태소 토큰화
# 한국어는 보편적으로 '형태로 분석기'로 토큰화 진행
# mecab사용
from konlpy.tag import Mecab
tokenizer=Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
print("Mecab 토큰화: ",tokenizer.morphs(kor_text))

# 문자 토큰화
print("문자 토큰화: ",list(en_text))







