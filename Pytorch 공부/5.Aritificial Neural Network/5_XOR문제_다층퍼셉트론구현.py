# XOR문제 - 다층 퍼셉트론 구현
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
# -------------------------------------------------------------
# PyTorch로 다층 퍼셉트론 구현
X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]]).to(device)
Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)

# 입력층, 은닉층1, 은닉층2, 은닉층3, 출력층
model = nn.Sequential(
    nn.Linear(2,10,bias=True), # input_layer=2, hidden_layer1=10
    nn.Sigmoid(),
    nn.Linear(10,10,bias=True), # hidden_layer1=10, hidden_layer2=10
    nn.Sigmoid(),
    nn.Linear(10,10,bias=True), # hidden_layer2=10, hidden_layer3=10
    nn.Sigmoid(),
    nn.Linear(10,1,bias=True), # hidden_layer3=10, ouput_layer = 1
    nn.Sigmoid()
).to(device)
# 비용함수, 옵티마이저 선언
criterion = torch.nn.BCELoss().to(device) # BCELoss() -> 이진분류 크로스엔트로피
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for epoch in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, cost.item())

# 학습된 다층 퍼셉트론의 예측값 확인
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값: ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값: ', predicted.detach().cpu().numpy())
    print('실제값: ', Y.cpu().numpy())
    print('정확도: ', accuracy.item())