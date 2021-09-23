import numpy as np
import torch
##### 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원 추가 #####
# 실습을 위해 (3,)의 크기 가지는 1인 차원텐서 생성
ft = torch.Tensor([0,1,2])
print("1인차원텐서 크기: ",ft.shape)
# 1번쨰 차원의 인덱스를 의미하는 숫자 0을 인자로 넣으면 1번쨰 차원에 1인 차원 추가
print("Unsqueeze: ",ft.unsqueeze(0)) #인덱스가 0부터 시작이므로 0은 1번쨰 차원의미
print("Unsqueeze의 크기: ",ft.unsqueeze(0).shape) #(1,3)의 2차원텐서로 변경확인
# (1,3)
##### 위 연산을 view로도 구현 가능 #####
# 2차원으로 바꾸고 싶으면서 1번째 차원이 1이면 view에서 (1,-1)로 지정하면됨
print("View로 구현:",ft.view(1,-1))
print("View로 구현크기: ",ft.view(1,-1).shape)

# unsqueeze의 인자값을 1로 넣어봄, 2번째 차원에 1을 추가하겠다는 의미
# (3,) -> (3,1)
print("unsqueeze 인자 1: ",ft.unsqueeze(1))
print("unsqueeze 인자 1의 크기: ",ft.unsqueeze(1).shape)
# (3,1)
# 인자로 -1을 넣으면 마지막 차원 의미 -> 현재는 (3,1)로 1넣은 거소가 동일 한 값이 나옴

## view(), squeeze(), unsqueeze()는 텐서의 원소수를 그대로 유지하며 모양과 차원조절