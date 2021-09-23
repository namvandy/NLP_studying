import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
# 모델 초기화
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w,b], lr=0.01)

epochs = 2000
for epoch in range(epochs + 1):
    # H(x)
    hypothesis= x_train * w + b
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch:{:4d} /{} W:{:.3f}, b:{:.3f} Cost:{:.6f}'.format(
            epoch, epochs, w.item(), b.item(), cost.item()
        ))

# 훈련결과
# 최적의 기울기 w는 2에 가까움
# b는 0에 가까움
# 실제정답은 H(x) = 2x 이므로 비슷하다.