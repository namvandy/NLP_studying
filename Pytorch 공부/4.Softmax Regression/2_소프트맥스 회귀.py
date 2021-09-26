# 소프트맥스 회귀(Softmax Regression)
# ---------------------------------------------------------------------
# 3개 이상의 선택지 중에서 1개를 고르는 다중클래스 분류를 풀기 위함

# 분류해야하는 클래스의 총 개수가 k일 때, k차원의 벡터를 입력받아 각 클래스에 대한 확률추정

# 소프트맥스 함수는 비용함수로 크로스 엔트로피 함수 사용
# 로지스틱회귀 함수에서의 크로스엔트로피함수와 구조적으로 동일함
# ---------------------------------------------------------------------

# 소프트맥스 회귀의 비용 함수 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

# PyTorch로 Softmax의 비용함수 구현하기(Low-Level)
z = torch.FloatTensor([1,2,3]) # 3개의 원소를 가진 벡터 텐서 정의
# 이 텐서를 softmax함수의 입력으로 사용
hypothesis = F.softmax(z, dim=0)
print("F.softmax: ",hypothesis) # tensor([0.0900, 0.2447, 0.6652]), 3개의 원소 값이 0~1사이의 값을 가지는 벡터로 변환
# 합이 1인지 확인
print("총 원소의 합: ",hypothesis.sum()) # tensor(1.) , 총 원소의 합: 1

# 비용 함수 직접 구현해보겠다.
z_1 = torch.rand(3,5, requires_grad=True) # 임의의 3x5 행렬
hypo = F.softmax(z_1, dim=1) # 각 샘플에 대해 소프트맥스 적용, 두번째 차원에 대해 적용(dim=1)
print("hypo: ",hypo) # 3개의 샘플에 대해 5개의 클래스 중 어떤 클래스가 정답인지 예측한 결과
# 각 샘플에 대한 임의의 레이블 생성
y = torch.randint(5,(3,)).long()
print("y: ",y)
# 각 레이블에 대해 원-핫 인코딩 수행
# 모든 원소가 0의 값을 가진 3x5 텐서 생성
y_one_hot = torch.zeros_like(hypo) # 모든 원소가 0인 3x5 텐서 생성
# .scatter_(dim, index, source)
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # y.unsqueeze(1) : (3,) -> (3,1) 텐서
# dim=1에 대해 수행, 세번쨰 인자 1로 y_unsqueeze(1)이 알려주는 위치에 1을 넣도록 함
# 연산 뒤에 _를 붙이면 Inplace=True(덮어쓰기연산)
print("y_one_hot: ",y_one_hot)
# cost(W) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{k}y_{j}^{(i)}\ log(p_{j}^{(i)})
# 비용함수 구현
cost = (y_one_hot * -torch.log(hypo)).sum(dim=1).mean()
print("cost: ",cost)

# ---------------------------------------------------------------------
# PyTorch로 Softmax의 비용 함수 구현(High-Level)
# ---------------------------------------------------------------------
# F.softmax() + torch.log() = F.log_softmax()

# 소프트맥스함수의 결과에 로그 씌울 때, 소프트맥스의 출려값을 로그함수의 입력으로 사용했음

# Low level : torch.log(F.softmax(z,dim=1))
# But, Pytorch에서는 2개의 함수를 결합한 F.log_softmax() 제공

# High level : F.log_softmax(z,dim=1)

# 비용함수
# F.log_softmax() + F.nll_loss() = F.cross_entropy()
# Low level  
# 1) (y_noe_hot * -torch.log(F.softmax(z,dim=1))).sum(dim=1).mean()
# 2) (y_one_hot * -F.log_softmax(z,dim=1).sum(dim=1).mean()

# High level 
# 1) F.nll_loss(F.log_softmax(z,dim=1), y) # 원-핫 벡터를 넣을 필요없이 실제값을 인자로 사용
# nll = Negative Log Likelihood
# nll_loss -> F.log_softmax()를 수행한 후 남은 수식들을 수행
# F.cross_entropy()는 F.log_softmax() + F.nll_loss()
# 2) F.cross_entropy(z,y)
# -> 비용 함수에 소프트맥스 함수까지 포함하고 있음을 기억