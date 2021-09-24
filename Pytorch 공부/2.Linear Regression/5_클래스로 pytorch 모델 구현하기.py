# class로 PyTorch 모델 구현
# -----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
# 단순 선형 회귀
# model = nn.Linear(1,1) 을 클래스로 구현
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 클래스
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
        # 단순 선형 회귀이므로 input_dim=1, output_dim=1
    def forward(self, x):
        return self.linear(x)
model_simple = LinearRegressionModel()

# 다중 선형회 귀
# model= nn.Linear(3,1) 을 클래스로 구현
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)
model_multi = MultivariateLinearRegressionModel()

# ---------------------------------------------------------
# 선형회귀모델 클래스 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model_linear = LinearRegression()
optimizer = torch.optim.SGD(model_linear.parameters(), lr=0.01)
epochs = 2000
for epoch in range(epochs+1):
    prediction = model_linear(x_train) #H(x)
    cost = F.mse_loss(prediction, y_train) # cost, MSE
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, cost.item()))
# --------------------------------------------------------------------
# 다중 선형회귀 클래스구현
x_mul_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_mul_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
class MultivariateLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)
model_multi_linear = MultivariateLinearRegression()
optimizer_multi_linear = torch.optim.SGD(model_multi_linear.parameters(), lr=1e-5)
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model_multi_linear(x_mul_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost_multi_linear = F.mse_loss(prediction, y_mul_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer_multi_linear.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost_multi_linear.backward()
    # W와 b를 업데이트
    optimizer_multi_linear.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost_multi_linear.item()))