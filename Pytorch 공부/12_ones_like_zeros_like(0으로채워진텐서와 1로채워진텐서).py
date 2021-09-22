##### ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서 #####
# (2x3) 텐서 생성
import torch

x = torch.FloatTensor([[0,1,2],[2,1,0]])
print("(2x3)텐서 x: ",x)
# 위 텐서에 ones_like -> 동일한 shape이지만 1로만 값이 채워진 텐서 생성
print("ones_like x: ",torch.ones_like(x))
# zeros_like -> 0으로만 값이 채워진 텐서 생성
print("zeros_like x: ", torch.zeros_like(x))