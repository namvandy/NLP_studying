# 과적합(Overfitting)을 막는 방법들

# 데이터의 양 늘리기 -> 데이터 증강
# -----------------------------------------------------------------------
# 모델의 복잡도 줄이기
import torch
from torch import nn

class Architecture(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Architecture, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out= self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
# 위 인공신경망이 입력데이터에 대해 과적합 현상 시
# 아래와 같이 인공신경망의 복잡도를 줄일 수 있음
class Architecture1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Architecture1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward1(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ---------------------------------------------------------------------

# 가중치 규제 적용하기 - Regularization

# L1 규제: 가중치 w들의 절대값 합계를 비용 함수에 추가
# L2 규제: 모든 가중치 w들의 제곱합을 비용 함수에 추가
# L1 규제 - 비용함수가 최소가 되게 하는 가중치, 편향을 찾는 동시에 가중치들의 절대값 합계도 최소가 되어야함
# -> 가중치 w의 값들이 0 또는 0에 가까이 작앚야 하므로 어떤 특성들은 거의 사용되지 않음
# L2 규제 - w의 값이 완전히 0이 되기보다는 0에 가까워지는 경향을 띔

# 어떤 특성들이 모델에 영향을 주고 있는지 판단 시 L1규제

# optimizer의 weight_decay 설정해 L2규제 적용
model = Architecture(10, 20, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
