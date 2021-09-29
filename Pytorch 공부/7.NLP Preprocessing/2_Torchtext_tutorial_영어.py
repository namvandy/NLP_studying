# Torchtext tutorial - 영어
# 텍스트에 대한 여러 추상화 기능 제공 라이브러리
# 직접 구현하는 것보다 사용하는 것이 더 편리하기도 함
'''
파일 로드하기(File Loading): 다양한 포맷의 corpus를 load
토큰화(Tokenization): 문장을 단어 단위로 분리
단어 집합(Vocab): 단어 집합을 만듬
정수 인코딩(Integer encoding): 전체 corpus의 단어들을 각각의 고유한 정수로 맵핑
단어 벡터(Word Vector): 단어 집합의 단어들에 대한 고유한 임베딩 벡터 생성. 랜덤값으로 초기화한 값일 수도 있고, 사전 훈련된 임베딩 벡터들을 로드할 수도 있음
배치화(Batching): 훈련 샘플들의 배치를 만들어줌. 이 과정에서 Padding도 이루어짐
'''
# 위 과정 이전/ 훈련,검증,테스트 데이터 분리 작업 별도
# 위 과정 이후/ 각 샘플에 대해 단어들을 임베딩 벡터로 맵핑해주는 작업(Lookup Table)은 nn.Embedding()을 통해 해결해야함.
# ------------------------------------------------------------------------
# IMDB 리뷰 데이터를 사용할 것임.
# 이전) from torchtext.data import TabularDataset
# 현재) from torchtext.legacy.data import TabularDataset

# 훈련 데이터, 테스트 데이터 분리
import urllib.request
import pandas as pd
# IMDB리뷰 데이터 다운
# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDB_Reviews.csv', encoding='latin1')
print(df.head(5))
print('전체 샘플의 수: {}'.format(len(df))) # 전체샘플: 50000
train_df = df[ : 25000]
test_df = df[25000 : ]
# train_df.to_csv("train_data.csv", index=False)
# test_df.to_csv("test_data.csv", index=False) # 인덱스 저장 X

# -------------------------------------------------------------

# 필드 정의(torchtext.data)
from torchtext.legacy import data
# 필드 정의
TEXT = data.Field(sequential=True, # 시퀀스 데이터 여부
                  use_vocab=True, # 단어집합을 만들 것인지
                  tokenize=str.split, # 어떤 토큰화 함수를 사용할 것인지
                  lower=True, # 영어 데이터를 전부 소문자화
                  batch_first=True, # 미니배치 차원을 맨 앞으로 해 데이터불러올것인지
                  fix_length=20) # 최대 허용길이, 이에 맞춰 패딩 작업 진행
LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True) # 레이블 데이터 여부
# 위 필드는 어떻게 전처리할 것인지 정의, 실제 데이터에 대해 진행 X

# -------------------------------------------------------------------

# 데이터셋 만들기
from torchtext.legacy.data import TabularDataset
# TabularDataset은 데이터를 불러오며 필드에서 정의했던 토큰화 방법으로 토큰화 수행
train_data, test_data = TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv',
    fields= [('text', TEXT), ('label', LABEL)], skip_header=True)
# path: 파일이 위치한 경로
# format: 데이터의 포맷
# fields: 위에 정의한 필드를 지정. (데이터셋내에서 해당필드 호칭할 이름, 지정할필드)
# skip_header: 데이터의 첫번째 줄 무시
print('훈련샘플의 수: {}'.format(len(train_data)))
print('테스트샘플의 수: {}'.format(len(test_data)))
# vars()를 통해 주어진 인덱스의 샘플 확인
print(vars(train_data[0]))
# print(train_data.fields.items()) #필드 구성 확인

# ----------------------------------------------------------------

# 단어집합 만들기
# .build_vocab()을 사용해 단어 집합 생성
TEXT.build_vocab(train_data, min_freq = 10, max_size=10000)
# min_freq: 단어 집합에 추가 시 단어의 최소 등장 빈도조건 추가
# max_size: 단어 집합의 최대 크기 지정
print('단어 집합의 크기: {}'.format(len(TEXT.vocab))) #10002개, 0번~10001번까지
# 임의로 특별 토큰 unk, pad추가, unk=단어집합에없는단어표현, pad=길이를맞추는패딩작업
print(TEXT.vocab.stoi) # 생성된 단어 집합 내의 단어들 확인

# ----------------------------------------------------------------

# Torchtext의 데이터로더 만들기
# 데이터로더는 미니배치만큼 데이터를 로드하게 만들어주는 역할
# Torchtext에서는 Iterator를 사용해 만듬
from torchtext.legacy.data import Iterator
batch_size= 5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)
print('훈련데이터 미니배치수: {}'.format(len(train_loader))) # 5000
print('테스트데이터 미니배치수: {}'.format(len(test_loader))) # 5000
# 25000개를 배치크기 5씩 묶어서 5000개

batch = next(iter(train_loader)) # 첫번째 미니배치
print(type(batch))
# 일반적 데이터로더는 미니배치를 텐서로 가져옴
# 토치텍스트의 데이터로더는 torchtext.data.batch.Batch 객체를 가져옴
# 실제 데이터 텐서에 접근하기 위해서는 정의한 필드명을 사용해야 함

# 첫번째 미니배치 text필드 호출
print(batch.text) # 배치크기가 5 -> 5개의 샘플 출력
# 각 샘플의 길이=20, 앞서 필드 정의시 fix_length=20 정해줬었음
# 하나의 미니배치크기 = (배치크기 X fix_length)
# 각 샘플은 더 이상 단어 시퀀스가 아닌 정수 시퀀스

