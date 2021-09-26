# 퍼셉트론(Perceptron)

# 단층퍼셉트론 - AND, NAND, OR 게이트 쉽게 구현 가능
def AND_gate(x1,x2):
    w1 = 0.5; w2=0.5; b=-0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
print("AND_gate: ",AND_gate(0,0), AND_gate(0,1), AND_gate(1,0), AND_gate(1,1))
def NAND_gate(x1, x2):
    w3=-0.5; w4=-0.5; b=0.7
    result = x1*w3 + x2*w4 + b
    if result <= 0:
        return 0
    else:
        return 1
print("NAND_gate: ",NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1))
def OR_gate(x1, x2):
    w5=0.6; w6=0.6; b=-0.5
    result = x1*w5 + x2*w6 + b
    if result <= 0:
        return 0
    else:
        return 1
print("OR_gate: ",OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1))

# 단층 퍼셉트론은 AND,NAND,OR 게이트를 구현할 수 있지만 XOR은 불가능
# XOR게이트는 입력값 두개가 서로 다른 값일 때 1,  서로 같은 값일 떄 0
# 선형 영역이 아니라 곡선, 비선형 영역으로 가능

# 다층퍼셉트론(MultiLayer Perceptron, MLP)
# 입력층 -> 은닉층 -> 출력층

# 은닉층이 2개 이상인 신경망을 심층 신경망(Deep Neural Network, DNN)