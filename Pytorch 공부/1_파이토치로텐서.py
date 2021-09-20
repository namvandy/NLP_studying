# Numpy로 텐서 만들기(벡터와 행렬 만들기)
import numpy as np
# 1차원: 벡터 2차원:행렬(Matrix) 3차원: 텐서(Tensor)
# 2차원텐서: |t| = (batch size, dim)
# 3차원텐서: |t| = (batch size, width, height)
# 자연어처리 3차원텐서 = (batch size, 문장길이, 단어벡터의차원)
# [숫자, 숫자, 숫자]와 같은 형식으로 만들고 np.array()로 감싸주면 됨.

# 1차원 with Numpy
t = np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)
print('Rank of t:', t.ndim) # 벡터의 차원 -> 1차원
print('Shape of t:',t.shape) # 벡터의 크기->(7,)=(1,7)의 크기를 가진 벡터
# 2차원 with Numpy
t_2 = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
print(t_2)
print('Rank of t_2:', t_2.ndim) # 벡터의 차원 -> 2차원
print('Shape of t_2:', t_2.shape) # 벡터의 크기 -> (4,3)=4행 3열

# PyTorch Tensor 선언하기
import torch
# 1차원 with PyTorch
p_1 = torch.FloatTensor([0,1,2,3,4,5,6])
print("p_1:",p_1)
print("p_1 차원:",p_1.dim()) # rank, 차원
print("p_1 shape:",p_1.shape)
print("p_1.size:",p_1.size())
# 현재 1차원 텐서이며, 원소는 7개
print(p_1[0],p_1[1],p_1[-1]) # tensor(0.) tensor(1.) tensor(6.)
# 2차원 with PyTorch
p_2 = torch.FloatTensor(
    [[1,2,3],
     [4,5,6],
     [7,8,9],
     [10,11,12]]
)
print("p_2:",p_2)
print("p_2.dim:",p_2.dim()) # 차원
print("p_2.size:",p_2.size()) # shape
# 현재 2차원 텐서이며, (4,3)의 크기를 가짐
print(p_2[:, 1]) # 1번쨰 차원전체선택, 2번째 차원의 첫번째 것만 가져옴
print(p_2[:, :-1]) # 1번째 차원전체선택, 2번째 차원에서 맨 마지막에서 첫번쨰 제외 모두
