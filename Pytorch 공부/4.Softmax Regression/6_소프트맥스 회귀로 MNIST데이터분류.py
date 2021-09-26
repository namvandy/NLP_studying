# 소프트맥스 회귀로 MNIST 데이터 분류
# MNIST 데이터는 숫자 0~9의 이미지롤 구성된 손글씨 데이터셋
# 각각의 이미지는 28픽셀 X 28픽셀 = 784픽셀
# 각 이미지를 784의 원소를 가진 벡터로 만들어줌. -> 총 784개의 특성을 가진 샘플이 됨.
'''
for X, Y in data_loader:
    # 입력 이미지를 [batch_size X 784]의 크기로 reshape
    # 레이블은 원-핫 인코딩
    X = X.view(-1, 28*28)
1. X는 for문에서 호출시 (배치크기 X 1 X 28 X 28)
2. view를 통해 (배치크기 X 784)로 변환
'''
# torchvision : 데이터셋, 구현된 모델들, 일반적 이미지 전처리 도구를 포함한 패키지
# torchtext : 자연어 처리 패키지
# -------------------------------------------------------------------------
# 분류긱 구현 위한 사전 설정
import torch
import torchvision.datasets as dsets
import  torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available() # GPU사용가능하면 True, 아니면 False
device = torch.device("cuda" if USE_CUDA else "cpu")
# print(device) #cuda
# 랜덤 시드 고정
random.seed(777)
torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
# 하이퍼파라미터를 변수로.
training_epochs = 15
batch_size = 100
# -------------------------------------------------------------------------
# MNIST 분류기 구현
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train = True,
                         transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)
# drop_last: 마지막 배치를 버릴 것인지
# 만약 1000개의 데이터에 배치크기 128이면, 마지막 104개가 남음. 이것을 버릴 때 사용
# 다른 미니배치보다 개수가 적은 마지막 배치를 경사하강법에 사용해 마지막배치가 상대적 과대평가를 막음
# 모델 설계
linear = nn.Linear(784,10, bias=True).to(device) # to()함수로 어디서 연산 수행할지
# 비용함수, 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적 softmax 포함
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost=0
    total_batch = len(data_loader)

    for X,Y in data_loader:
        # 배치크기:100 -> 아래의 연산에서 X는 (100,784)의 텐서
        X = X.view(-1, 28*28).to(device)
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0~9 정수
        Y = Y.to(device)
        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    print('Epoch:', '%04d' % (epoch+1), 'cost = ', '{:.9f}'. format(avg_cost))
print('Learning Finished')

# 테스트 데이터를 사용해 모델 테스트
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()