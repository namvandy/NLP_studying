# Torchtext tutorial - 한국어
import urllib.request
import pandas as pd
# 네이버영화리뷰데이터 다운
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')
print(train_df.head())
# id, document, label 3개의 열로 구성
# document - 영화리뷰, label - 긍정,부정, id는 불필요
print('훈련데이터 샘플수: {}'.format(len(train_df)))
print('테스트데이터 샘플수: {}'.format(len(test_df)))

# ----------------------------------------------------------------------
# 필드 정의하기(torchtext.data)
from torchtext.legacy import data
from konlpy.tag import Mecab
tokenizer = Mecab(dicpath='C:/mecab/mecab-ko-dic')
# 데이터가 3개의 열로 구성 -> 3개의 필드정의
ID = data.Field(sequential=False,
                use_vocab=False) # 실제사용X
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs,
                  lower=True,
                  batch_first=True,
                  fix_length=20)
LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True)
# 필드정의만 진행, 어떤 전처리도 하지 않음

# ----------------------------------------------------------------------
# 데이터셋 만들기
from torchtext.legacy.data import TabularDataset
# 데이터셋의 형식 변경, 동시에 토큰화 수행
train_data ,test_data = TabularDataset.splits(
    path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
    fields=[('id', ID),('text', TEXT),('label', LABEL)], skip_header=True)
print("훈련 샘플의 수: {}".format(len(train_data)))
print("테스트 샘플의 수: {}".format(len(test_data)))

print(vars(train_data[0]))

# -------------------------------------------------------------------
# 단어집합(Vocabulary) 만들기
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print("단어집합의크기: {}".format(len(TEXT.vocab)))
# print(TEXT.vocab.stoi)

# ---------------------------------------------------------------------
# Torchtext의 데이터로더 만들기
from torchtext.legacy.data import Iterator
batch_size=5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)

batch = next(iter(train_loader)) # 첫번째 미니배치
print(batch.text)
# 미니 배치의 크기 = (배치 크기 × fix_length)