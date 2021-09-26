# nn.Module로 구현하는 로지스틱회귀
# nn.Linear() + nn.Sigmoid() = 로지스틱회귀의 가설식
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
# 훈련데이터 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
# nn.Sequential() -> nn.Modul층을 차례로 쌓게 함
# Wx + b 와 같은 수식과 시그모이드 함수 등과 같은 여러 함수들을 연결해주는 역할.
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid() # 출력은 시그모이드함수를 거친다.
)
print("훈련데이터를 넣은 예측값: ",model(x_train)) # w,b는 랜덤초기화된 상태
# (6x1) 텐서. 현재 w와 b는 임의의 값을 가지므로 현재예측은 의미가 없음

# 경사하강법 사용해 훈련
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

epochs = 1000
for epoch in range(epochs + 1):
    # H(x) 계산
    hypothesis = model(x_train)
    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, epochs, cost.item(), accuracy * 100,
        ))
print("훈련 후의 w와 b의 값: ", list(model.parameters()))