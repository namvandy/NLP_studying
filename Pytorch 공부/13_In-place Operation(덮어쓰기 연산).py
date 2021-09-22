##### In-place Operation(덮어쓰기 연산) #####
import torch
x = torch.FloatTensor([[1,2],[3,4]]) # (2x2) 텐서
print("곱하기 2를 한 값: ",x.mul(2,))
print("기존 값:", x)
# 곱하기 한 결과를 x에다 다시 저장하지 않았으니 기존 값은 변하지 않음

# 연산 뒤에 _를 붙이면 기존의 값을 덮어쓰기함
print("_를 추가해 쓴 값:",x.mul_(2,)) # 결과를 x에 저장하며 결과를 출력
print("_를 추가한 후 기존 값:", x )
