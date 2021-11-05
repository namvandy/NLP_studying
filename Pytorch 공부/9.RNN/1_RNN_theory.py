# 순환 신경망(Recurrent Neural Network, RNN)
# Sequence 모델 . 입력과 출력을 시퀀스 단위로 처리.

# 이전 신경망들은 전부 은닉층에서 활성화 함수를 지난 값을 출력층 방향으로 향함. -> Feed Forward 신경망
# RNN은 은닉층 활성화 함수를 지난 값을 출력층에도 보내면서 다시 은닉층 노드의 다음 계산의 입력으로 보냄.
# RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할의 노드를 cell이라고 칭함. -> 이전의 값을 기억하려하는 일종의 메모리 역할 수행 -> 메모리 셀 or RNN 셀 표현
# 메모리 셀은 각갂의 time step에서 바로 이전 시점에서의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동 진행
# 현재 시점: t .-> t에서의 메모리 셀이 가진 값은 과거의 메모리 셀들에 영향을 받은 것을 의미.
# 메모리 셀이 출력층 or 다음 시점 t+1의 자신에게 보내는 값을 은닉상태(hidden state). -> t시점의 메모리 셀은 t-1시점의 메모리 셀이 보낸 은닉상태값을 t기점의 은닉상태 계산을 위한 입력값으로 사용

# RNN은 일대다(one-to-many), 다대일(many-to-one), 다대다(many-to-many)등 입력과 출력의 길이를 다르게 설계 가능
# RNN 셀의 각 시점 별 입,출력의 단위는 보편적이게 '단어벡터'임.

# one-to-many는 하나의 이미지 입력에 대해 사진의 제목을 출력하는 이미지 캡셔닝 작업에 사용가능 -> 사진의 제목: 단어들의 나열 -> 시퀀스 출력
# many-to-one은 감성 분류, 스팸 메일 분류에 사용 가능
# many-to-many는 챗봇, 번역기, 개체명 인식이나 품사 태깅과 같은 작업이 속함.

# 현재시점:t, 현재시점의 은닉상태값 h_t
# h_t는 입력층의 입력값을 위한 가중치 W_x, 이전 시점 t-1의 은닉상태값 h_(t-1)를 위한 가중치 W_h 를 가짐
# 은닉층: h_t = tanh(W_x * x_t + W_h * h_(t-1) + b)
# 출력층: y_t = f(W_y * h_t + b)    ->  f는 비선형 활성화 함수 중 하나

# 실제 구현에 앞서 의사 코드(pseudocode)작성 -> 실제 동작코드 아님
'''

hidden_state_t = 0 # 초기 은닉상태 초기화
for input_t in input_length: # 각 시점에 대해 입력받음
    output_t = tanh(input_t, hidden_state_t) # 각 시점의 입력, 은닉 상태로 연산
    hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉상태

'''

# RNN 층을 실제 동작되는 코드 구현. //// (timesteps, input_size) 크기의 2D텐서를 입력받았다고 가정 -> 실제 파이토치에서는 (batch_size, timesteps, input_size) 크기의 3D 텐서 입력받음
import numpy as np
timesteps = 10 # 시점의 수, 자연어에서는 보통 문장의 길이
input_size = 4 # 입력의 차원. 자연어에서는 보통 단어 벡터의 차원
hidden_size = 8 # 은닉상태의 크기. 메모리 셀의 용량

inputs = np.random.random((timesteps,input_size)) # 입력에 해당되는 2D 크기 텐서
hidden_state_t = np.zeros((hidden_size, )) # 초기은닉상태 0 초기화 # 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬
print(hidden_state_t) # 8의 크기를 가지는 은닉 상태. 현재 모두 0을 가짐

# 가중치, 편향 정의
Wx = np.random.random((hidden_size, input_size)) # (8,4) 2D 텐서 생성. 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # (8,8) 2D 텐서 생성. 은닉상태에 대한 가중치
b = np.random.random((hidden_size, )) # (8, )크기의 1D 텐서 생성. 편향
print("입력가중치크기:", np.shape(Wx))
print("은닉가중치크기:", np.shape(Wh))
print("편향크기:", np.shape(b))

# 모든 시점의 은닉 상태 출력 가정. RNN층 동작
total_hidden_states = []
# 메모리셀 동작
for input_t in inputs:
    output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states)) # 각 시점 t별 메모리셀의 출력의 크기는 (timestep, output_dim)
    hidden_state_t = output_t
total_hidden_states = np.stack(total_hidden_states, axis=0) # 출력 시 값을 깔끔히 정리
print("total_hidden_states:", total_hidden_states)

# ---------------------------------------------------------------------------------------------------------------------

# PyTorch의 nn.RNN()
import torch
import torch.nn as nn
torch_input_size = 5
torch_hidden_size = 8
torch_inputs = torch.Tensor(1, 10 , 5) # 입력 텐서(배치크기 X 시점의수 X 매시점의 입력) (batch_size, time_steps, input_size)
cell = nn.RNN(torch_input_size, torch_hidden_size, batch_first=True) # batch_first=True -> 입력텐서의 첫번째 차원이 배치 크기임을 알려줌
outputs, _status = cell(torch_inputs)
# RNN 셀은 두개의 입력 리턴 -> 첫번쨰: 모든 시점(timesteps)의 은닉 상태들. 두번째: 마지막 시점(timesteps)의 은닉 상태
print("모든 time-step의 hidden_state:",outputs.shape) # 10번의 시점동안 8차원의 은닉상태가 출려됨
print("최종 time-step의 hidden_state", _status.shape) # (1,1,8)의 크기