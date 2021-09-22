import numpy as np
import torch
##### 스퀴즈(Squeezee) - 1인 차원을 제거한다. #####
# (3x1) 의 크기를 가지는 2차원텐서를 생성해 실습
ft = torch.FloatTensor([[0],[1],[2]])
print("(3x1) 의 크기를 가지는 2차원텐서: ",ft)
print("(3x1) 의 크기를 가지는 2차원텐서 크기: ",ft.shape)
# 두번째 차원이 1이므로 squeeze를 사용하면 (3,)크기의 텐서로 변경
print("(3x1) 의 크기를 가지는 2차원텐서 squeeze: ",ft.squeeze())
print("(3x1) 의 크기를 가지는 2차원텐서 squeeze의 크기: ",ft.squeeze().shape)