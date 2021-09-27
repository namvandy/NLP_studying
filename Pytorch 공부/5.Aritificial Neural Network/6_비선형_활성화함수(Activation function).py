# 비선형 활성화 함수(Activation function)
# 입력을 받아 수학적 변환을 수행하고 출력을 생성하는 함수
# 은닉층에서 왜 활성화 함수로 시그모이드를 지양해야하는지
# 은닉층에서 주로 사용되는 ReLU 함수
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------

# 활성화 함수의 특징 - 비선형 함수
# 선형 함수로는 은닉층을 여러번 추가하더라도 1회 추가한 것과 차이를 줄 수 없음

# ---------------------------------------------------------------

# Sigmoid function과 기울기 소실
# 인공신경망 - 입력 -> 순전파연산 -> 예측값과 실제값의 오차를 손실함수-> 손실을 미분해 기울기 -> 역전파연산
# Sigmoid function의 문제는 미분을 해서 기울기를 구할 때 발생
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0,5.0,0.1)
y=  sigmoid(x)
plt.plot(x,y)
plt.plot([0,0],[1.0,0.0], ":")
plt.title('Sigmoid fucntion')
plt.show()
# 출력값이 0 또는 1에 가까워지면 기울기가 완만해짐 -> 기울기 계산 시 0에 가까운 아주 작은 값
# -> 역전파과정에서 0에 가까운 아주 작은 기울기 곱해질 시 기울기 잘 전달 안됨. -> 기울기 소실(Vanishing Gradient)문제
# Sigmoid function을 사용하는 은닉층의 개수가 다수 -> 0에 가까운 아주 작은 기울기 계속 곱해짐 -> W 업데이트불가 -> 학습 X

# ---------------------------------------------------------------

# 하이퍼볼릭탄젠트 함수(Hyperbolic tangent function)
# 입력값을 -1~1사이로 반환
x_tanh = np.arange(-5.0,5.0,0.1)
y_tanh = np.tanh(x)
plt.plot(x_tanh,y_tanh)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh function')
plt.show()
# 하이퍼볼릭탄젠트 함수도 시그모이드 함숭와 같은 문제 발생
# 하지만 다르게 0을 중심으로 하고 있음 -> 반환값의 변화폭이 더큼 -> 기울기 소실증상이 시그모이드보다 적은 편

# ---------------------------------------------------------------

# 렐루 함수(ReLU)
# f(x) = max(0,x)
def relu(x):
    return np.maximum(0,x)
x_relu = np.arange(-5.0,5.0,0.1)
y_relu = relu(x_relu)
plt.plot(x_relu,y_relu)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()
# 음수 입력 -> 0, 양수 입력 -> 입력값 그대로 반환
# 특정 양수값에 수렴하지 않은 깊은 신경망에서 좋음
# 입력값이 음수이면 기울기도 0 -> 회생 어려움 -> 죽은 렐루(dying ReLU)

# ---------------------------------------------------------------

# 리키 렐루(Leaky ReLU)
# 죽은렐루를 보완하기 위한 변형 함수
# 입력값이 음수일 경우에 0이 아니라 0.001같은 매우 작은 수 반환
# f(x) = max(ax,x)
a= 0.1
def leaky_relu(x):
    return np.maximum(a*x, x)
x_leaky_relu = np.arange(-5.0,5.0,0.1)
y_leaky_relu = leaky_relu(x_leaky_relu)
plt.plot(x_leaky_relu,y_leaky_relu)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky Relu Function')
plt.show()
# 입력값이 음수라도 기울기가 0이 되지 않으면 ReLU는 죽지 않음

# ---------------------------------------------------------------

# 소프트맥스 함수(Softmax function)
# 분류 문제를 로지스틱 회귀와 소프트맥스 회귀를 출력층에 적용해 사용
x_softmax = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y_softmax = np.exp(x) / np.sum(np.exp(x_softmax))

plt.plot(x_softmax, y_softmax)
plt.title('Softmax Function')
plt.show()
# 시그모이드, 소프트맥스는 출력층의 뉴런에서 주로 사용
# 시그모이드 -> 이진분류, 소프트맥스 -> 다중분류

# ---------------------------------------------------------------

# 출력층의 활성화 함수와 오차함수의 관계

'''
   문제 - 활성화 함수 - 비용 함수
이진분류 - 시그모이드 - nn.BCELoss()
다중클래스분류 - 소프트맥스 - nn.CrossEntropyLoss()
   회귀 - 없음 - MSE
'''