##### nn.Module로 구현하는 선형 회귀 #####
# PyTorch에 이미 구현되어 있는 함수들로 선형회귀모델을 구현
# 선형회귀모델 : nn.Linear()
# 평균제곱오차 : nn.functional.mse_loss()
'''
함수의 사용 예제
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)
import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)
'''
# -------------------------------------------------------
# 단순 선형회귀 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
# 데이터 선언 / y=2x를 가정한 상태에서 만들어진 데이터. 이미 정답은 w=2,b=0
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
# nn.Linear() 은 입력의 차우너, 출력의 차원을 인수로 받음
# 모델을 선언 및 초기화. 단순선형회귀이므로 input_dim=1, output_dim=1
model = nn.Linear(1,1)
# model은 가중치w와 편향b가 저장되어 있음 / model.parameters()로 호출가능
print("model의 w,b: ", list(model.parameters()))

# optimizer 설정. model.parameters()를 사용해 w,b 전달
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 2000번 반복
epochs = 2000
for epoch in range(epochs+1):
    # 가설 선언
    prediction = model(x_train)
    # 비용함수
    cost = F.mse_loss(prediction, y_train)
    # 비용함수로 가설 개선
    optimizer.zero_grad() # gradient 0으로 초기화
    cost.backward() # 비용함수를 미분해 gradient 계산
    optimizer.step() # w,b 업데이트
    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()))

# cost값이 매우 작게 나옴. w,b 도 최적화 되었는지 확인
# x에 임의의 값 4를 넣어 확인
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var) # forward 연산
# 8에 가까운 값이 나와야 제대로 된것
print("훈련 후 4일때 예측값: ", pred_y)
print("학습 후의 w,b: ", list(model.parameters()))
# H(x)식에 입력 x로부터 예측된 y얻는 것을 forward 연산
# 비용함수를 미분해 기울기를 구하는 것을 backward 연산

# -----------------------------------------------------------