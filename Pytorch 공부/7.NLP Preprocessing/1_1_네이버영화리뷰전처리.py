# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Vocabulary 생성
# 네이버 영화 리뷰 분류하기 데이터 사용 - 20만개의 영화 리뷰를 긍정1, 부정0으로 레이블링한 데이터
import urllib.request
import pandas as pd
from IPython.core.display import display
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')
display(data[:10])
print('전체 샘플의 수 : {}'.format(len(data)))
sample_data = data[:100] # 임의의 100개 저장
# 정규 표현식 사용해 데이터 정제
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
display(sample_data[:10])
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
tokenized= []
for sentence in sample_data['document']:
    temp = tokenizer.morphs(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized.append(temp)
print(tokenized[:10])
# 빈도수 계산 도구 FreqDist()
vocab = FreqDist(np.hstack(tokenized))
print("단어 집합의 크기: {}".format(len(vocab)))
# 단어를 key로 , 빈도수를 value로 저장되어 있음.
# vocab에 단어 입력 시 빈도수를 리턴
# vocab['재밌'] -> 10
# most_common()는 상위 빈도수를 가진 주어진 수의 단어만을 리턴 -> 등장 빈도수가 높은 단어들을 원하는 개수만큼 얻을 수 있음
# 등장 빈도수 상위 500개의 단어만 단어집합으로 저장
vocab_size = 500
voca = vocab.most_common(vocab_size)
print("단어 집합의 크기: {}".format(len(vocab)))

# 각 단어에 고유한 정수 부여
# 인덱스 0,1은 다른 용도로 남기고 나머지 단어들 2~501 인덱스로 부여
word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []
for line in tokenized: # 입력데이터에서 1줄씩
    temp = []
    for w in line: # 각 줄에서 1개씩 글자를 읽음
        try:
            temp.append(word_to_index[w]) # 글자를 해당되는 정수로 변환
        except KeyError: #단어 집합에 없는 단어일 경우 unk로 대체
            temp.append(word_to_index['unk'])
    encoded.append(temp)
print(encoded[:10])

# 패딩: 길이가 다른 문장들을 동일 길이로 변환
max_len = max(len(l) for l in encoded)
print('리뷰의 최대 길이: %d' % max_len)
print('리뷰의 최소 길이: %d' % min(len(l) for l in encoded))
print('리뷰의 평균 길이: %f' % (sum(map(len,encoded))/len(encoded)))
plt.hist([len(s) for s in encoded], bins=50)
plt.xlabel('length of sample')
plt.ylabel('number of sample')
plt.show()
# 가장 긴 길이는 62, 62로 길이 통일
for line in encoded:
    if len(line) < max_len:
        line += [word_to_index['pad']] * (max_len - len(line)) # 나머지는 전부 'pad' 토큰으로 채움
print('리뷰의 최대 길이 : %d' % max(len(l) for l in encoded))
print('리뷰의 최소 길이 : %d' % min(len(l) for l in encoded))
print('리뷰의 평균 길이 : %f' % (sum(map(len, encoded))/len(encoded)))
print(encoded[:3])

# 고유한 정수로 맵핑 -> 각 정수를 고유한 단어벡터로 바꾸는 작업 필요
# 단어벡터를 얻는 법 -> 원-핫 인코딩, 워드 임베딩
# 주로 워드 임베딩 사용
