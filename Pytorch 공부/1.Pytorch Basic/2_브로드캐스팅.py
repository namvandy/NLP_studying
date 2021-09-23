# 브로드캐스팅(Broadcatsing)
# 두 행렬 A,B 존재 -> 덧셈,뺄셈을 하려면 두 행렬의 크기가 같아야함
# 곱셈을 하려면 A의 마지막 차원과 B의 첫번째 차원의 일치해야함
# 브로드캐스팅: 자동으로 크기를 맞춰 연산을 수행
import torch

m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2) #m1, m2의 크기는 둘다 (1,2)이여서 가능

# 크기가 다른 텐서들 간의 연산
# 벡터와 스칼라가 덧셈 연산을 수행
# Vector + scalar
m3 = torch.FloatTensor([[1,2]])
m4 = torch.FloatTensor([3]) # [3] -> [3,3]
print(m3+m4)
# m3은 (1,2) m4는 (1,) -> pytorch가 m4를 (1,2)로 변경해 연산 수행

# 벡터 간 연산에서 브로드캐스팅 적용
# 2x1 vector + 1x2 vector
m5 = torch.FloatTensor([[1,2]])
m6 = torch.FloatTensor([[3],[4]])
print(m5+m6)
# [1,2] -> [[1,2],[1,2]]
# [3],[4] -> [[3,3],[4,4]]
