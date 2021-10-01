# 워드 임베딩(Word Embedding)
# 단어를 벡터로 표현
# 워드 임베딩: 단어를 밀집 표현으로 변환

# -------------------------------------------------------------
# 희소 표현(Sparse Representation)
# 벡터 또는 행렬(matrix)의 값이 대부분이 0으로 표현되는 방법
# -> 원-핫 벡터는 희소 벡터임
# 문제점 : 단어의 개수가 증가 -> 벡터의 차원이 한없이 커짐
import torch
# 원-핫 벡터 생성
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])
# 원-핫 벡터간 코사인 유사도
print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))
# -> 단어 간 유사도를 구할 수 없음

# ---------------------------------------------------------

# 밀집 표현(Dense Representation)
# 벡터의 차원을 단어집합의 크기로 상정하지 않음
# 사용자설정 값으로 모든 단어의 벡터표현의 차원을 맞춤
# 0,1이 아닌 실수값 가짐
# 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0]
# 이 때 1 뒤의 0의 수는 9995개. 차원은 10,000
# 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...] -> 밀집벡터
# 이 벡터의 차원은 128

# -----------------------------------------------------
# 워드 임베딩
# 워드 임베딩 과정 통해 나온 결과 = 임베딩 벡터
# 워드 임베딩 방법: LSA, Word2Vec, FastText, Glove
# nn.embedding()은 단어를 랜덤한 값의 밀집벡터로 변환 -> 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습

'''
	             원-핫 벡터	       임베딩 벡터
차원|	    고차원(단어 집합의 크기)	저차원
다른 표현|	희소 벡터의 일종	     밀집 벡터의 일종
표현 방법|	수동	                 훈련 데이터로부터 학습함
값의 타입|	1과 0	             실수
'''