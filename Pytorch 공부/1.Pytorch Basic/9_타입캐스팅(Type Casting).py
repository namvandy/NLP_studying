##### 타입캐스팅(Type Casting) #####
# 텐서에는 자료형이 존재 , 각 데이터형별로 정의됨
# e.g. 32비트의 부동소수점은 torch.FloatTensor
# e.g. 64비트의 부호있는정수는 torch.LongTensor
# e.g. GPU연산을 위한 자료형은 torch.cuda.FloatTensor
# 이 자료형을 변환하는 것을 타입 캐스팅이라고 정의
# 실습을 위해 long타입의 lt텐서 선언
# https://wikidocs.net/images/page/52846/newimage.png
import torch

lt = torch.LongTensor([1,2,3,4])
print("long type tensor: ",lt)
# .float()을 붙이면 바로 float형 변경
print("long type tensor float형: ",lt.float())

# Btype의 bt텐서 선언
bt = torch.ByteTensor([True,False,False,True])
print("bt: ",bt)
# .long() -> long 타입 텐서, .float() -> float 타입 텐서
print("bt long type: ",bt.long())
print("bt float type: ",bt.float())