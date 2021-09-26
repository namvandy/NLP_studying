# 로지스틱 회귀(Logistic Regression)
# 이진분류(Binary Classification) 풀기 위한 대표적인 알고리즘
# 이름은 회귀이지만 분류작업에 사용 가능

# -------------------------------------------------------

# 이진분류(Binary Classification)
'''
---------------------
|score(x) | result(y)|
|45       |   불합격  |
|50       |   불합격  |
|55       |   불합격  |
|60       |   합격    |
|65       |   합격    |
|70       |   합격    |
---------------------
합격 1, 불합격 0
'''
# 0과 1들로 표현하는 그래프는 알파벳의 S자 형태로 표현이 됨.
# 이러한 관계를 표현하기위해서는 Wx + b같은 직선함수가 아니라 S자형태로 표현가능한 함수 필요
# 직선을 사용할 경우 분류 작업이 잘 동작하지 않음
# 로지스틱 회설의 가설은 Wx + b가 아니라 특정함수 f를 추개하 f(Wx + b)의 가설사용
# S자 형태 그릴 수 있는 f는 시그모이드 함수

# ---------------------------------------------------------

# 시그모이드 함수(Sigmoid function)
# H(x) = sigmoid(Wx + b) = 1 / 1 + e ** (-Wx+b)
# 식 구현

import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1+np.exp(-x))
# W가 1, b가 0인 그래프
'''
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()
'''
# 선형 회귀에서는 W가 직선의 기울기를 의미.
# 여기서는 그래프의 경사도 의미 / W의 값이 커지면 경사가 커지고, 작아지면 경사가 작아미
# b값의 변화에 따른 그래프 좌,우 이동

# Sigmoid 함수는 0~1사이의 값 가짐. 출력값이 0.5이상 True, 0.5이하 False
# 입력값이 한없이 커지면 1에 수렴, 입력값이 한없이 작아지면 0에 수렴

# ---------------------------------------------------------

# 비용함수(Cost function)
# H(x) = sigmoid(Wx+b)의 최적의 W와 b를 찾을 수 있는 비용함수 정의해야함.
# 선형회귀에서 사용했던 MSE를 사용하면 심한 비볼록형태의 그래프가 나옴
# Sigmoid에 MSE를 적용했을 때 경사하강법 사용할 경우의 문제점:
# 오차가 최솟값이 되는 구간에 도착했다고 판단되는 구간이 실제 오차최솟값이 되는 구간이 아닐 수도 있다.
# 다른 구간이 최솟값이 될 수 있음 -> 잘못 판단할 경우 최적의 W를 찾지못해 성능이 오르지 않음
# y의 실제값이 1 / if y=1 -> cost(H(x), y) = -log(H(x))
# y의 실제값이 0 / if y=0 -> cost(H(x), y) = -log(1-H(x))
# 하나의 식으로 통합
# cost(H(x), y) = -[ylogH(x) + (1-y)log(1-H(x))]

# ---------------------------------------------------------
# 로지스틱 회귀 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data) # (6,2)
y_train = torch.FloatTensor(y_data) # (6,1)
# XW 성립 위해서는 W는 (2x1)크기이어야 함
w = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = 1 / (1+ torch.exp(-(x_train.matmul(w) + b)))
print("예측값: ",hypothesis)
# y_train과 크기가 동일한 (6x1)크기의 예측값 벡터가 나오는데 모두 0.5
# PyTorch에서는 시그모이드 제공
# hypothesis = torch.sigmoid(x_train.matmul(w) + b)
print("실제값: ",y_train)
# 모든 원소에 대해 오차 구하기
losses = -(y_train * torch.log(hypothesis) +
           (1-y_train) * torch.log(1-hypothesis))
print("모든원소의오차: ",losses)
# 전체 오차에 대한 평균
cost = losses.mean()
print("오차평균: ",cost)

# PyTorch에서는 로지스틱회귀의 비용함수를 이미 구현해서 제공함.
# import torch.nn.functional as F
# F.binary_cross_entropy(hypothesis, y_train)
print("pytorch지원 로지스틱비용함수: ",F.binary_cross_entropy(hypothesis, y_train))

# -----------------------------------------------------
# 전체모델
w = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([w,b], lr=1)
epochs=1000
for epoch in range(epochs+1):
    hypothesis = torch.sigmoid(x_train.matmul(w)+b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    #cost = cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))
print("훈련후의 값: ",hypothesis) # 현재 w와 b는 훈련후의 값을 가짐 -> 출력
prediction = hypothesis >= torch.FloatTensor([0.5]) #0.5넘으면True
print("0,1분류: ",prediction)
print("훈련후의 W: ",w)
print("훈련후의 b: ",b)