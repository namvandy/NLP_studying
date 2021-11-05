import torch

# L1 거리
# L1 norm 사용, 맨해튼거리라고도 함.
def L1_distance(x1,x2):
    return ((x1-x2).abs()).sum()
# L2 거리
# 유클라디안 거리
def L2_distance(x1,x2):
    return ((x1-x2)**2).sum()**.5
# L infinity 거리
# infinity norm
def L_infinity_distance(x1,x2):
    return ((x1-x2).abs()).max()
# 코사인 유사도
# 벡터차원의 크기가 클수록 연산량 부담
# 희소벡터일 경우 윗변이 벡터의곱 -> 0이 들어간 차원이 많으면 해당 차원이 직교하며 곱의 값이 0 -> 정확한 유사도 or 거리 측정 불가
def cosine_similarity(x1,x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)
# 자카드 유사도
# 두 집합 간의 유사도 측정  #윗변=교집합, 밑변=합집합
def jaccard_similarity(x1,x2):
    return torch.stack([x1,x2]).min(dim=0)[0].sum() / torch.stack([x1,x2]).max(dim=0)[0].sum()
