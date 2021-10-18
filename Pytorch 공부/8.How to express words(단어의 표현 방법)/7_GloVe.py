# 글로브(GloVe) : Global Vectors for Word Representation
# 카운트 기반, 예측 기반을 모두 사용
# 기존 카운트 기반의 LSA(Latent Semantic Analysis), 예측기반 Word2Vec의 단점을 보완
# LSA :  카운ㅌ트 기반 행렬 입력 -> 차원 축서 -> 잠재된 의미 도출
# Word2Vec : 실제값과 예측값에 대한 오차를 손실 함수를 통해 줄여나가며 학습하는 예측기반

# LSA는 카운트 기반으로, corpus의 전체적 통계정보고려 / 단어의미의 유추 작업에는 성능 떨어짐
# Word2Vec은 예측기반으로 단어 간 유추 작업에 뛰어남 / 임베딩 벡터가 주변단어를 고려하기 때문에 전체적인 정보 반영 불가능

# -------------------------------------------------------------------------------

# 윈도우 기반 동시 등장 행렬(Window based Co-occurrence Matrix)
# 단어의 동시 등장 행렬은 행과 열을 전체 단어집합의 단어들로 구성.
# i 단어의 윈도우크기 내 k 단어 등장 횟수를 i행 k열에 기재한 행렬
# 전치(Transpose)해도 동일한 행렬 진행 -> i단어의 윈도우크기 내 k단어 빈도 = k단어 윈도우 크기 내 i단어 등장 빈도

# -------------------------------------------------------------------------------

# 동시 등장 확률(Co-occurrence Probability)
# P( k | i ) : 동시등장확률. 특정 단어 i의 전체 등장횟수 카운트, 특정단어 i 등장 시 어떤 단어 k의 등장횟수 카운트
# i를 중심단어, k를 주변단어라 했을 때, 중심 단어 i의 행의 모든 값을 더한 값을 분모
# i행 k열의 값을 분자로 한 값

# GloVe의 아이디어 = 임베딩 된 중심 단어와 주변 단어 벡터의 내적이 전체 코퍼스에서의 동시 등장 확률이 되도록 만드는 것
# X_ij : 중심 단어 i가 등장 시 윈도우 내 주변 단어 j의 등장횟수
# 손실함수 : X_ik의 값이 작으면 상대적으로 함수의 값은 작게, 값이 크면 함수의 값은 상대적으로 크게
# X_ik가 지나치게 높다해서 지나친 가중치 주지 않기위해 함수의 최댓값은 1

# -------------------------------------------------------------------------------
# pip install glove==1.0.0
# GloVe 훈련
# 위 word2vec 영어 버전의 코드를 불러옴
import nltk
nltk.download('punkt')
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

targetXML = open('ted_en-20160408.xml','r',encoding='UTF8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
content_text = re.sub(r'\([^)]*\)', '',parse_text)
sent_text = sent_tokenize(content_text)
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    # 각 문장에 대해 구두점 제거, 대문자->소문자
    normalized_text.append(tokens)
result=[]
result = [word_tokenize(sentence) for sentence in normalized_text]
from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
model_result = model.wv.most_similar("man")
loaded_model = KeyedVectors.load_word2vec_format('eng_w2v')
# -----------------------------------------------------------
from glove import Corpus, Glove
corpus = Corpus()
corpus.fit(result, window=5) # 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# 학습 완료

model_result = glove.most_similar("man") # 입력단어의 가장 유사한 단어리스트 리턴
print(model_result)