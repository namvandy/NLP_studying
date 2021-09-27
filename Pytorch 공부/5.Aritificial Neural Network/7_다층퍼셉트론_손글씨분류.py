# 다층 퍼셉트론으로 손글씨 분류
# 각 이미지는 0~15의 명암을 가지는 8X8=64 픽셀 해상도의 흑백이미지
# 1797개

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.images[0]) # 첫번째 샘플
# print(digits.target[0]) # 첫번째 샘플의 레이블 -> 0
print('전체 샘플의 수: {}'.format(len(digits.images)))
# 상위 5개의 샘플만 시각화
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):
    plt.subplot(2,5,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('Greys'), interpolation='nearest')
    plt.title('sample: %i' % label)
# 상위 5개 샘플의 레이블 확인
for i in range(5):
    print(i,'번 인덱스샘플 레이블: ',digits.target[i])
# 훈련데이터와 레이블을 X,Y에 저장
# digits.images -> 8X8 행렬
# digits.data -> 8X8 행렬을 전부 64차원의 벡터로 변환해 저장
X = digits.data
Y = digits.target

# -----------------------------------------------------------------

# 다층 퍼셉트론 분류기 만들기
import torch
import torch.nn as nn
from torch import optim

model = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10),
)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

    losses.append(loss.item())

plt.plot(losses)