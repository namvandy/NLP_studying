# 미니배치와 데이터로드(Mini Batch and Data Load)
# 데이터를 로드하는 방법, 미니배치 경사하강법
# ------------------------------------------------------
import torch

# 미니배치와 배치크기
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 행렬로 묶어 처리하였지만 이것보다 적은 단위로 나누어 학습 -> 미니배치(Mini Batch)
# 미니배치만큼만 가져가 비용을 계산, 경사하강법 수행. -> 다음 미니배치를 가져가서 수행, 반복
# 이러헤 전체 데이터에 대한 학습이 1회 끝나면 1epoch
# 전체 데이터에 대해 한번에 경사하강법 수행 = '배치경사하강법'
# 미니 배치 단위로 경사하강법 수행 = '미니배치 경사하강법'
# 배치경사하강법은 전체데이터를 사용 -> 가중치 값이 최적값에 수렴하는 과정이 안정적 / 계산량이 많이듬
# 미니배치경사하강법은 전체데이터의 일부만 보고 사용 -> 최적값 수렴과정 값이 헤메지만 속도 빠름
# 배치크기는 보통 2의 제곱수=> CPU & GPU 의 메모리가 2의 배수이므로 2의 제곱수일 때 데이터 송수신 효율 높일 수 있음

# --------------------------------------------------------
# 이터레이션(Itreation)

# 이터레이션은 한번의 epoch내에 이루어지는 매개변수인 W, b의 업데이트 횟수
# 전체 데이터:2000, 배치크기:200 -> 이터레이션:10

# --------------------------------------------------------
# 데이터 로드(Data Load)
# PyTorch 제공 데이터다루기 도구 -> 데이터셋(DAtaset), 데이터로더(DataLoader)
# 이를 사용해 미니배치학습, 데이터셔플, 병렬처리 수행가능
# Dataset을 정의하고 DataLoader에 전달하는 방식으로 사용
# Dataset은 커스텀해서 생성가능
# 여기서는 텐서를 입력받아 Dataset의 형태 변환 -> TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset #텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
# TensorDataset은 텐서를 입력으로 받음. tensor형태로 데이터를 정의
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
# TensorDataset의 입력으로 사용, dataset으로 저장
dataset = TensorDataset(x_train,y_train)
# dataset을 만듦 -> DataLoader 사용가능
# DataLoader는 기본적으로 2개의 인자 -> 데이터셋, 미니배치의 크기 / shuffle도 많이 사용
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# 훈련
epochs = 100
for epoch in range(epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        # print("batch_idx: ",batch_idx)
        # print("samples: ",samples)
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))
# 임의의 입력
new_var = torch.FloatTensor([[73,80,75]])
pred_y = model(new_var)
print("훈련후 입력이 73,80,75의 예측값: ", pred_y)