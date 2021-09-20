# 행렬 곱셈과 곱셈의 차이
# Matrix Multiplication vs Multiplication
# 행렬로 곱셈 => 행렬곱셈(matmul) & 원소별 곱셈(mul)
# PyToch tensor의 행렬곱셈 => matmul()을 통해 수행
import torch
import numpy as np
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print("행렬1의 shape:",m1.shape) # -> 2x2 행렬
print("행렬2d의 shape:",m2.shape) # -> 2x1 행렬(벡터)
print(m1.matmul(m2)) # 2 x 1
# 위 결과는 2x2행렬과 2x1행렬(벡터)의 행렬 곱셈의 결과

# element-wise 곱셈이라 는 것이 존재
# 동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱하는 것
# 아래는 서로 다른 크기의 행렬이 브로드캐스팅 후 element-wise 곱셈 수행 -> * 또는 mul()을 통해 수행됨
m3 = torch.FloatTensor([[1,2],[3,4]])
m4 = torch.FloatTensor([[1],[2]])
print(m3*m4)
print(m3.mul(m4))
