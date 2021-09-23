# 뷰(View) - 원소의 수를 유지하며 텐서의 크기 변경
# PyTorch의 View = Numpy의 reshape
import numpy as np
import torch
t = np.array([[[0,1,2],
              [3,4,5]],
              [[6,7,8],
              [9,10,11]]]) # 3차원 텐서 생성
ft = torch.FloatTensor(t)
print(ft.shape) # (2,2,3)

##### 3차원 텐서 -> 2차원 텐서로 변경 #####
# ft를 view를 사용해 변경
print(ft.view([-1,3])) #ft라는 텐서를 (?,3)의 크기로 변경
print(ft.view([-1,3]).shape)
# view( [-1,3] ) -> -1의 첫번째 차원은 사용자가 잘 모르겠으니 파이토치에게 맡기겠다.
# 3은 두번째 차원의 길이는 3을 가져라
# 3차원 텐서를 2차원 텐서로 변경하되 (?,3)의 크기로 변환해라
# (2,2,3) -> (2x2 , 3) -> (4,3) 으로 크기 변환

##### 3차원 텐서의 크기 변경 #####
# 3차원 텐서로 차원은 유지하되, 크기(shape)을 바꾸는 작업
# view로 텐서의 크기를 변경하더라도 원소의 수는 유지되어야 함.
# (2x2x3) -> (?x1x3) => ?는 몇차원인가? Answer=4
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)

