# 소프트맥스 회귀 구현
# Low-level과 F.cross_entropy를 사용해 구현
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
# 훈렫 데이터와 레이블 선언
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]  # 각 샘플은 4개의 특성 가짐, 총 8개의 샘플존재
y_train = [2, 2, 2, 1, 1, 1, 0, 0] # 0,1,2를 가짐 -> 총 3개의 클래스 존재
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# ---------------------------------------------------------------------
# 소프트맥스 회귀 구현(Low-level)
print("x_train.shape: ",x_train.shape) #torch.Size([8, 4])
print("y_train.shape: ",y_train.shape) #torch.Size([8])
# 최종 사용할 레이블은 y_train에서 원-핫 인코딩을 한 결과여야 함.
# -> 클래스가 3개이므로 y_train에서 원-핫 인코딩 결과는 8x3이어야함.
y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print("y_one_hot.shape: ",y_one_hot.shape) # torch.Size([8,3) -> W행렬은 4x3이어야함

w = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([w,b], lr=0.1)

epochs=1000
for epoch in range(epochs+1):
    hypothesis = F.softmax(x_train.matmul(w)+b,dim=1)
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))
print("="*100)
# ---------------------------------------------------------------------
# 소프트맥스 회귀 구현(High-level)
# F.cross_entropy()를 사용해 비용함수 구현
w_1 = torch.zeros((4,3), requires_grad=True)
b_1 = torch.zeros(1,requires_grad=True)

optimizer_1 = optim.SGD([w_1,b_1], lr=0.1)
epochs_1=1000
for epoch in range(epochs_1+1):
    z = x_train.matmul(w_1) + b_1
    cost_1 = F.cross_entropy(z, y_train)
    optimizer_1.zero_grad()
    cost_1.backward()
    optimizer_1.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs_1, cost_1.item()
        ))
