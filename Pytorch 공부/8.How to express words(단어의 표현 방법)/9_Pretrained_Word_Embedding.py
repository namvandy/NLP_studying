# 사전 훈련된 워드 임베딩(Pretrained Word Embedding)
# 이미 훈련된 워드 임베딩을 불러 임베딩 벡터로 사용
# 훈련 데이터가 부족하다면 nn.Embedding보단 사전훈련된 임베딩 벡터 호출이 나은 선택
# -----------------------------------------------------------------------

# IMDB 리뷰 데이터를 훈련 데이터로 사용
from torchtext import datasets
from torchtext.legacy import data,datasets # 이걸로해야되네...Field가 정의안됬었음
# 두 개의 필드 객체 정의
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
print("훈련 데이터의 크기:{}".format(len(trainset)))
# 첫번째 샘플
# print(vars(trainset[0]))

# torchtext의 field 객체의 build_vocab을 통해 사전 훈련된 워드 임베딩 사용가능
# 직접 훈련시킨 사전훈련된 워드 임베딩 사용방법 / torchtext제공 사전훈련된 워드 임베딩 사용방법
# ------------------------------------------------------------------------------
# torchtext를 사용한 사전 훈련된 워드임베딩

# 사전 훈련된 Word2Vec 모델 확인
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')
# print(word2vec_model['this']) # 'this'의 임베딩 벡터값 출력
# print(word2vec_model['self-indulgent']) # 학습시 존재 X해서 에러 발생

# 사전훈련된 Word2Vec을 초기 임베딩 사용
import torch
import torch.nn as nn
from torchtext.vocab import Vectors
vectors = Vectors(name="eng_w2v") # eng_w2v 모델을 vectors에 저장(사전훈련된 워드투벡터 모델)
TEXT.build_vocab(trainset, vectors=vectors, max_size=10000, min_freq=10) # Word2Vec 모델을 임베딩 벡터값으로 초기화
# max_size: 단어집합의 크기 제한, min_freq: 등장 빈도수
# vectors: 만들어진 단어집합의 각 단어의 임베딩 벡터값으로 env_w2v에 저장되었던 임베딩 벡터값들로 초기화

# print(TEXT.vocab.stoi) # 현재 단어 집합의 단어와 맵핑된 고유한 정수 출력 # 10002개의 단어 존재
#<unk>, <pad>는 특별 토큰이므로 실제 10000개 존재
print('임베딩 벡터의 개수와 차원:{}'.format(TEXT.vocab.vectors.shape))
# print(TEXT.vocab.vectors[0]) # <unk>의 임베딩 벡터값, 0으로 초기화 된 상태
# print(TEXT.vocab.vectors[1]) # <pad>의 임베딩 벡터값, 0으로 초기화 된 상태
# print(TEXT.vocab.vectors[10]) # this의 임베딩 벡터값, env_w2v에 저장된 'this'의 임베딩벡터값과 동일
# print(TEXT.vocab.vectors[10000]) # self-indulgent의 임베딩 벡터값, env_w2v에 존재하지 않았기에 0으로 초기화 됨
# 이 임베딩 벡터들을 nn.Embedding()의 초기화 입력으로 사용
embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
# print(embedding_layer(torch.LongTensor[10])) # 단어 this의 임베딩 벡터값

# -----------------------------------------------------------------------
# torchtext에서 제공하는 pre-trained word embedding
'''
제공되는 임베딩 벡터 리스트의 일부
fasttext.en.300d
fasttext.simple.300d
glove.42B.300d
glove.840B.300d
glove.twitter.27B.25d
glove.twitter.27B.50d
glove.twitter.27B.100d
glove.twitter.27B.200d
glove.6B.50d
glove.6B.100d
glove.6B.200d
glove.6B.300d
'''
# IMDB 리뷰데이터에 존재하는 단어들을 torchtext제공 pre-train embedding vector값으로 초기화
from torchtext.vocab import GloVe
TEXT.build_vocab(trainset, vectors=GloVe(name='6B',dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(trainset)
print('임베딩 벡터의 개수와 차원:{}'.format(TEXT.vocab.vectors.shape))
# 이 임베딩 벡터들이 저장된 TEXT.vocab.vectors를 nn.Embedding()의 초기화 입력으로 사용
embedding_layer2 = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
print(embedding_layer2(torch.LongTensor([10]))) # this의 임베딩 벡터값