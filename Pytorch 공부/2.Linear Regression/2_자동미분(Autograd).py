##### 자동미분(Autograd) #####

# 경사하강법 코드를 보면 requires_grad=True, backward() 등이 나옴.
# 이는 pytorch에서 제공하는 자동미분기능을 수행하고 있는 것
# 모델이 복잡해질수록 경사하강법을 넘파이 등으로 코딩하는 것은 까다로움 -> pytorch는 자동미분지원
# 자동미분으로 미분계산을 자동화해 경사하강법을 손쉽게 사용
import torch
# 값이 2인 스칼라 텐서 w 선언
w = torch.tensor(2.0, requires_grad=True) # 텐서에 대한 기울기를 저장하겠다.
# 수식 정의
y = w ** 2
z = 2 * y + 5
# 해당 수식을 w에 대해 미분 / .backward()를 호출해 해당 수식의 w에 대한 기울기계산
z.backward()
print('수식을 w로 미분한 값: {}'.format(w.grad))