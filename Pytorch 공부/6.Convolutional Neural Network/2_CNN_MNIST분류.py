# CNN으로 MNIST 분류하기
# 첫번째 표기방법
# 합성곱(nn.Cov2d) + 활성화함수(nn.ReLU)를 하나의 합성곱층
# 맥스풀링(nn.MaxPoold2d)은 풀링층
# 두번째 표기항법
# 합성곱 + 활성화함수 + 맥프숲핑을 하나의 합성곱 층으로 봄

# 두번째 표기방법을 택해서 구성
'''
# 1번레이어: 합성곱층
합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1)
+ 활성화 함수 ReLU + 맥스풀링(kernel_size=2, stride=2)
# 2번레이어: 합성곱층
합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1)
+ 활성화 함수 ReLU + 맥스풀링(kernel_size=2, stride=2)
# 3번레이어: 전결합층
특성맵을 펼친다. -> batch_size X 7 X 7 X 64 -> batch_size X 3136
전결합층(뉴런10개) + 활성화 함수 Softmax
'''
# ----------------------------------------------------------------------
# 모델 구현
import torch
import torch.nn as nn
# 임의의 텐서(배치크기 x 채널 x 높이 x 너비)
inputs = torch.Tensor(1, 1, 28, 28)
print('Tensor의 크기: {}'.format(inputs.shape))
# 합성곱층과 풀링 선언
# 첫번째 합성곱 층
conv1 = nn.Conv2d(1, 32, 3, padding=1) # 1채널입력, 32채널출력, 커널3, 패딩 1
# 두번째 합성곱 층
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32채널입력, 64출력, 커널3, 패딩1
# 맥스풀링
pool = nn.MaxPool2d(2) # 2를 인자,  커널사이즈와 스트라이드가 둘 다 해당값으로 지정
# 구현체를 연결해 모델 만들기
out = conv1(inputs) #torch.Size([1, 32, 28, 28])
out = pool(out) #torch.Size([1, 32, 14, 14)
out = conv2(out) #torch.Size([1, 64, 14, 14]) # 패딩을 1, 3x3커널을 사용하면 크기가 보존됨.
# 맥스풀링
out = pool(out) #torch.Size([1, 64, 7, 7])

# 이제 이 텐서를 펼치는 작업
# 텐서의 n번째 차원에 접근하게 해주는 .size(n)
out.size(0) # 1
out.size(1) # 64
out.size(2) # 7
out.size(3) # 7
# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1)
print("펼친 out 크기: ",out.shape) # torch.Size([1, 3136])
# 배치 차원 제외 모두 하나의 차원으로 통합됨.
# 전결합층 통과 -> 출력층으로 10개의 뉴런 배치 -> 10개 차원의 텐서로 변환
fc = nn.Linear(3136,10)
out = fc(out)
print("전결합층 통과 out크기: ", out.shape)

# ----------------------------------------------------------------------

# CNN으로 MNIST 분류
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import TensorDataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train = True,
                          transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                          train = False,
                          transform=transforms.ToTensor(),
                          download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
# 모델 설계
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 두번째층
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 전결합층 7x7x64 inputs -> 10 ouputs
        self.fc = torch.nn.Linear(7*7*64, 10, bias=True)
        # 전결합층 한정 가중치 초기화
        torch.nn.init.xavier_uniform(self.fc.weight)
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0), -1)
        out=self.fc(out)
        return out
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device) #비용함수에 소프트맥스포함
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {} '.format(total_batch))
# 총배치수=600, 배치크기=100 -> 훈련 데이터는 총 60000개
# 훈련
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in data_loader: #미니배치단위로. X는 미니배치, Y는 레이블
        # 이미지는 이미 (28,28) -> no reshape
        # 라벨은 원-핫 인코딩 되지 않음
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, avg_cost))


# 테스트
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())