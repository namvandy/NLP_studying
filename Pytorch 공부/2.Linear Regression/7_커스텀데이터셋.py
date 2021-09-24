# 커스텀 데이터셋(Custom Dataset)
# torch.utils.data.Dataset을 상속받아 커스텀데이터셋 만드는 경우 존재
# torch.utils.data.Dataset은 PyTorch에서 데이터셋을 제공하는 추상클래스

# ----------------------------------------------------
# 커스텀 데이터셋의 기본적인 뼈대.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CUstomDataset(torch.utils.data.Dataset):
    def __init__(self): ...
    # 데이터셋의 전처리 부분
    def __len__(self): ...
    # 데이터셋의 길이. 즉, 샘플의 수 적는 부분
    def __getitem__(self, idx): ...
    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수 부분

# ----------------------------------------------------
# 커스텀 데이터셋으로 선형회귀 구현
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self): # 총 데이터의 개수 리턴
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
epochs=20
for epoch in range(epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
# 임의의 입력
new_var = torch.FloatTensor([[74,80,75]])
pred_y = model(new_var)
print("훈련후 임의의입력의 예측값: ", pred_y)