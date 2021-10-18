# PyTorch의 nn.Embedding()
# 임베딩 벡터의 사용 방법 두가지
# 1) 임베딩 층을 만들어 훈련데이터로부터 처음부터 임베딩 벡터를 학습
# 2) 미리 사전에 훈련된 임베딩 벡터를 가져와 사용하는 방법

# 임베딩 층은 look-up table
# 임베딩 층의 입력으로 사용 시 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어야함.
# 어떤 단어 -> 단어에 부여된 고유 정수값 -> 임베딩 층 통과 -> 밀집 벡터
# 임베딩 층은 입력 정수에 대해 밀집벡터로 맵핑
# 밀집벡터는 가중치가 학습되는 것과 같은 방식으로 훈련 -> 밀집벡터 = 임베딩벡터
# word -> integer -> lookup table -> embedding vector
# 임베딩 층의 입력이 원-핫 벡터가 아니어도 동작.
# 단어->정수인덱스->원-핫벡터->입력으로 사용 X ==>> 단어->정수인덱스->입력으로 사용-> 룩업테이블의 결과인 임베딩 벡터 리턴
# ---------------------------------------------------
# 임의의 문장생성, 단어집합생성, 각 단어에 정수 부여
import torch

train_data = 'you need to know how to code'
word_set = set(train_data.split()) # 중복제거 단어집합 생성
vocab = {word: i+2 for i, word in enumerate(word_set)} # 각 단어에 고유 정수 맵핑
vocab['<unk>']=0
vocab['<pad>']=1
print(vocab)

# 단어 집합의 크기를 행으로 가지는 임베딩 테이블 구현. 차원은 3으로 정함.
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])
# 임의의 문장에 대해 룩업테이블을 통해 임베딩 벡터 가져옴
sample = 'you need to run'.split()
idxes = []
# 각 단어 정수 변환
for word in sample:
    try:
        idxes.append(vocab[word])
    except KeyError: # 단어 집합에 없는 단어일 경우 <unk>로 대체
        idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

lookup_result = embedding_table[idxes, :] # 각 정수를 인덱스로 임베딩 테이블에서 값 가져옴
print(lookup_result)
# ---------------------------------------------------------------------

# 임베딩 층 사용하기 / nn.Embedding()으로 사용
data = 'you need to know how to code'
set_word = set(data.split())
vocabs = {tkn: i+2 for i,tkn in enumerate(word_set)}
vocabs['<unk>']=0
vocabs['<pad>']=1
import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings= len(vocabs), # 단어집합의 크기
                               embedding_dim=3, #임베딩 할 벡터의 차원
                               padding_idx=1) # 패딩을 위한 토큰의 인덱스 알려줌
print(embedding_layer.weight)
