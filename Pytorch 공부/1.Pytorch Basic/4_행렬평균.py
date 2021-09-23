import numpy as np
import torch
# 평균 -> numpy에서의 사용법과 매우 유사
# 1차원 벡터 선언 -> .mean()을 사용해 원소의 평균
t = torch.FloatTensor([1,2])
print(t.mean()) # tensor(1.5000)
# 2차원 벡터 선언 -> .mean()
t2 = torch.FloatTensor([[1,2],[3,4]])
print(t2)
print(t2.mean()) # => 4개의 원소의 평균인 2.5가 나옴
# dim=0 -> 차원을 인자로(dim=0 -> 첫번째 자원)
print(t2.mean(dim=0))
print(t2.mean(dim=1))
'''
1)dim=0
[[1,2],
 [3,4]] 에서 1과3의 평균, 2와4의 평균 -> [2, 3]
2)dim=1
[[1,2],
 [3,4]] 에서 1,2의 평균, 3,4의 평균 -> [1.5, 3.5]
'''
# dim=0 -> 행제거
# dim=1 -> 열제거
# dim= -1 -> 열제거