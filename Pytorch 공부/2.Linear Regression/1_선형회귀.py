##### 선형회귀이론, PyTorch로 선형회귀모델 구현 #####
# 데이터에 대한 이해 - 학습할 데이터에 대해 알아봄
# 가설 수립 - 가설 수립 방법
# 손실 계산 - 학습 데이터로 모델 개선시킬 때 손실(loss) ㅣ용
# 경사하강법 - 학습을 위한 핵심 알고리즘

# ------------------------

# 기본셋팅
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1) # random seed -> random_state같은 것

# -------------------------

# 변수 선언
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
print("x_train:{}\n x_train.shape:{} ".format(x_train, x_train.shape))
print("y_train:{}\n y_train.shape:{} ".format(y_train, y_train.shape))
# x_train, y_train -> (3x1)

# --------------------------

# 가중치와 편향의 초기화
# 선형회귀 -> 학습데이터와 가장 잘 맞는 하나의 직선을 찾는 것
# 가장 잘 맞는 직선 -> W , b -> W, b의 값을 찾는 것
# W를 0으로 초기화하고 학습통해 값이 변경되는 변수임을 명시시
w = torch.zeros(1, requires_grad=True) # requires_grad: 학습을 통해 계속 같이 변경되는 변수임을 의미
# 가중치 w를 출력
print("가중치: ",w) # 가중치 w가 0으로 초기화되어있어 0이 출력
# b도 마찬가지로 진행
b = torch.zeros(1, requires_grad=True)
print("편향: ",b)
# 현재는 어떤 값이 들어가도 가설은 0을 얘측  y= 0*x + 0

# -------------------------

# 가설세우기
# 직선의 방정식에 해당되는 가설 선언
# H(x) = Wx + b
hypothesis = x_train * w + b
print("가설 식: ",hypothesis)

# -------------------------

# 비용함수 선언하기
# 선형회귀의 비용함수에 해당되는 평균제곱오차(MSE) 선언
cost = torch.mean((hypothesis - y_train) ** 2)
print("비용함수: ", cost)

# -------------------------

# 경사하강법 구현하기
# SGD는 경사하강법의 일종, lr은 학습률
optimizer = optim.SGD([w,b], lr=0.01)
# gradient를 0으로 초기화 -> 초기화해야 새로운 가중치 편향에 대해 새로운 기울기 가능
optimizer.zero_grad()
# 비용함수를 미분해 gradient계산
cost.backward() # 가중치 w와 편향 b에 대한 기울기가 계산이 됨
# w와 b를 업데이트
optimizer.step() # w와 b에서 리턴되는 변수들의 기울기에 lr=0.01을 곱하여 뺴줌으로 업데이트

# -------------------------

# optimizer.zero_grad()가 필요한 이유
# pytorch는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있음

i = torch.tensor(2.0, requires_grad=True)
epochs = 20
for epoch in range(epochs+1):
    z = 2 * i
    z.backward()
    print('수식을 i로 미분한 값: {}'.format(i.grad))
# 계속해서 미분값인 2가 누적됨.
# 그렇기 때문에 optimizer.zero_grad()로 미분값을 계속 0으로 초기화시켜주어야 함.