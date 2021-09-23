##### 다중선형회귀(MUltivariable Linear regression)  #####
# 다수의 x로부터 y를 예측
# 앞에서 한 선형회귀는 단순 선형회귀(Simple Linear Regression)

# -----------------------------------------

# 독립 변수 x의 개수가 3개. / 3개의 퀴즈 점수로부터 최종 점수를 예측하는 모델 생성
'''
Quiz1 (x1) | Quiz(x2) | Quiz(x3) | Final(y) |
    73     |    80    |     75   |    152   |
    93     |    88    |     93   |    185   |
    89     |    91    |     80   |    180   |
    96     |    98    |    100   |    196   |
    73     |    66    |     70   |    142   |
'''
# 수식: H(x) = w1 * x1 + w2 * x2 + w3 * x3 + b

# -----------------------------------------

# PyTorch로 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# 훈련 데이터
x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[66]])
x3_train = torch.FloatTensor([[75],[93],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# 가중치 w와 편향 b 선언
w1 = torch.zeros(1,requires_grad=True)
w2 = torch.zeros(1,requires_grad=True)
w3 = torch.zeros(1,requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설, 비용함수, 옵티마이저 선언, 경사하강법 1000회 반복
optimizer = optim.SGD([w1,w2,w3,b], lr=1e-5)
epochs= 1000
for epoch in range(epochs+1):
    # 가설 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    # 비용함수 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # 비용함수로 가설 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
print("="*50)
# -----------------------------------------

# 벡터와 행렬 연산으로 바꾸기
# x의 개수가 많아질 경우 x와 w를 일일이 선언해주어야 함. 가설식도 마찬가지. => 비효율적
# -> 행렬 곱셉 연산(또는 벡터의 내적) 사용해서 해결
# 식을 벡터로 표현하면 변수로 압축가능 -> 다수의 샘플의 병렬연산이 되어 속도증가

# -----------------------------------------
# 행렬 연산을 고려해 PyTorch로 구현

a_train = torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])  # 하나에 모든 샘플을 선언했음(이전엔 3개를 선언)
b_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
print("a.train shape: ",a_train.shape) # (5x3)
print("b.train shape: ",b_train.shape) # (5x1)

# 가중치와 편향 선언
w_1 = torch.zeros((3,1), requires_grad=True) # a.train이 (5x3)이니까 (3x1)로 선언해 행렬곱 가능하게 선언
b_1 = torch.zeros(1, requires_grad=True)
# 가설 선언
hypothesis_1 = a_train.matmul(w_1) + b_1
# optimizer 선언
optimizer_1 = optim.SGD([w_1,b_1], lr=1e-5)
new_epoch = 20
for epoch in range(new_epoch+1):
    # H(x)계산, 가설선언
    # 편향 b_1 는 브로드캐스팅되어 각 샘플에 더해짐
    hypothesis_1 = a_train.matmul(w_1) + b_1
    # 비용함수계산
    cost_1 = torch.mean((hypothesis_1 - b_train) ** 2)
    # 비용함수로 가설 개선
    optimizer_1.zero_grad()
    cost_1.backward()
    optimizer_1.step()
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, new_epoch, hypothesis_1.squeeze().detach(), cost_1.item()
    ))