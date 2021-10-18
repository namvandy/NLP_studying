# 영어/한국어 Word2Vec 훈련
# gensim패키지에 Word2Vec 구현되어 있음
# -------------------------------------------------

# 영어 Word2Vec
import nltk
nltk.download('punkt')
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
# 훈련 데이터:  https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip
# 데이터 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
# 해당 파일은 xml문법으로 작성됨 -> 전처리 필요 , 실질적 데이터는 <content> </content> 사이
# 사이의 내용중 배경음 나타내는 단어도 등장 -> 제거

# 훈련데이터 전처리
targetXML = open('ted_en-20160408.xml','r',encoding='UTF8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content> </content> 사이의 내용만 가져옴

content_text = re.sub(r'\([^)]*\)', '',parse_text)
# sub모듈로 content중간 (Audio), (Laughter) 등의 배경으음 제거
# 해당 코드는 괄호로 구성된 내용을 제거하는 것

sent_text = sent_tokenize(content_text) #입력 코퍼스에 대해 NLTK로 문장 토큰화 진행

normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    # 각 문장에 대해 구두점 제거, 대문자->소문자
    normalized_text.append(tokens)
result=[]
result = [word_tokenize(sentence) for sentence in normalized_text]
# 단어 토큰화 진행

print("총샘플수:{}".format(len(result)))
for line in result[:3]:
    print(line) # 상위 3개로 토큰화 확인

# Word2Vec 훈련
from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

# size=워드벡터의 특징값. 즉, 임베딩 된 벡터의 차원
# window= context window size
# min_count= 단어 최소 빈도 수 제한
# workers= 학습을 위한 프로세스 수
# sg = 0은 CBOW, 1은 Skip-Gram
# 입력한 단어에 대해 가장 유사한 단어 출력 = model.wv.most_similar
model_result = model.wv.most_similar("man")
print(model_result)
# 단어의 유사도 계산 가능

# Word2Vec 모델 저장, 로드하기
model.wv.save_word2vec_format('./eng_w2v') # 모델저장
loaded_model = KeyedVectors.load_word2vec_format('eng_w2v') # 모델로드
model_result = loaded_model.most_similar("man")
print(model_result)