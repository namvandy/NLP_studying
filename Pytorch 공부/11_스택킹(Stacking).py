##### 스택킹(Stacking) ######
# 연결을 하는 또 다른 방법
# concatenate하는 것보다 더 편리할 때 존재 -> stacking은 많은 연산을 포함
# 실습을 위해 크기가 (2,)인 동일한 3개의 벡터 생성
import torch

x= torch.FloatTensor([1,4])
y= torch.FloatTensor([2,5])
z= torch.FloatTensor([3,6])
# torch.stack을 통해 3개의 벡터 스택킹
print("x,y,z stacking: ",torch.stack([x,y,z]))
# 3개의 벡터가 순차적으로 쌓여 (3x2)의 텐서 변경

# 스택킹의 연산축약 -> 아래 코드는 위와 동일한 연산 작업임
print("x,y,z unsqueeze, cat: ",torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
# unsqueeze(0)으로 (2,) -> (1,2)로 변경 -> 연결(cat)으로 (3x2) 텐서로 변경

# stacking에 dim 인자 줄수 있음
# dim=1  -> 두번쨰 차원이 증가하게 쌓아라
print("x,y,z stacking dim=1: ",torch.stack([x,y,z],dim=1))