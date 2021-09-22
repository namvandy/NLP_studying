##### 연결하기(concatenate) #####
import torch

x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
# 두 텐서를 torch.cat([]) 을 통해 연결
# torch.cat은 어느 차원을 늘릴 것인지 인자로 줄수 있음, dim=0은 첫번째 차원 늘리라는 의미
print("x,y concat, dim=0: ",torch.cat([x,y],dim=0))
# (2x2),(2,2) -> (2x4)
print("x,y concat, dim=1: ",torch.cat([x,y],dim=1))
# (2x2),(2x2) -> (4x2)
