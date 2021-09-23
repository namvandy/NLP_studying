# max -> 원소의 최대값 리턴
# argmax -> 최대값을 가진 인덱스 리턴
# (2,2)크기 행렬 선언
import torch

t = torch.FloatTensor([[1,2], [3,4]])
print("max:",t.max())
print("dim=0, max:",t.max(dim=0))
#-> 행의 차원을 제거해 (1,2)텐서를 만들고 결과는 [3,4]
# indices=tensor([1,1]) -> argmax도 같이 리턴됨.
# 첫번째 열 3의 인덱스는 1, 두번째 열 4의 인덱스는 1
# [ [1,2], [3,4] ] 에서
# 첫번째 열 0번 인덱스=1 , 1번 인덱스=3
# 두번째 열 0번 인덱스=2 , 1번 인덱스=4
# => 3과 4의 인덱스는 [1,1]
# max또는 argmax만 리턴받고 싶다면
print('max:',t.max(dim=0)[0])
print('armax:',t.max(dim=0)[1])