# 원-핫 인코딩(One-Hot Encoding)
# 범주형 데이터 처리할 떄 레이블을 표현하는 방법
# 선택지의 개수만큼 차원을 가지며, 각 선택지의 엔득세 해당하는 원소에 1, 나머지에 0의 값을 가지도록하는 표현방법
# 선택지 3개 / 강아지, 고양이, 냉장고
'''
강아지 = [1,0,0]
고양이 = [0,1,0]
냉장고 = [0,0,1]
원-핫 인코딩이 된 벡터 - > 원-핫 벡터(One-Hot Vector)
'''
#정수 인코딩
'''
정수 인코딩의 순서 정보가 도움이 되는 분류문제도 있음
각 클래스가 순서의 의미를 갖고 있어 회귀 통해 분류 가능한 경우
{baby,child,adolescent,adult}
{1층,2층,3층,4층}
{10대,20대,30대,40대} 같은 경우
But, 일반적 분류 문제에서는 순서의미를 가지지 않음. -> 각 클래스 간의 오차는 균등한 것이 옳음
'''
'''
원-핫 인코딩은 분류 문제 모든 클래스 간의 관계를 균등하게 분배함
모든 클래스에 대해 원-핫 인코딩을 통해 얻은 원-핫 벡터들은 
모든 쌍에 대해 유클리드 거릴르 구해도 전부 동일.
BUt, 원-핫 벡터의 관계의 무작위성은 때로는 단어의 유사성을 구할 수 없는 단점으로 언급됨.
'''