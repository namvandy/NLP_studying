# 머신러닝 용어

# 전체데이터 = 훈련 + 검증 + 테스트
# 분류, 회귀
# 지도학습, 비지도학습, 강화학습
# 샘플: 하나의 데이터, 하나의 행, 특성: 종속변수 y를 예측하기 위한 각각의 독립변수 x
# 혼동행렬
'''
|   |  참  | 거짓 |
|참 |  TP  | FN  |
|거짓| FP  | TN   |
TP(True Positive)  예측-진실, 정답-True
FN(False Negative) 예측-거짓, 정답-False
FP(False Positive) 예측-진실, 정답-False
TN(True Negative): 예측-거짓, 정답-True

# 정밀도(Precision) : 양성이라고 대답한 전체 케이스에 대한 TP의 비율
정밀도 = TP / (TP+FP)
# 재현율(Recall) : 실제값이 양성인 데이터의 전체 개수에 대해 TP의 비율
재현율 = TP / (TP+FN)
'''
# 과대적합 & 과소적합