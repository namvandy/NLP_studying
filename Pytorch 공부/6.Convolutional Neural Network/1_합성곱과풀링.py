# 합성곱과 풀링(Convolution and Pooling)

# CNN 은 이미지 처리에 탁월한 성능
# 합성곱층과 풀링층으로 구성
# 합성곱층 = CONV(합성곱연산) + ReLU(활성화 함수)
# 풀링층 = POOL(풀링연산)

# 합성곱층은 합성곱연산을 통해 이미지의 특징을 추출
# 커널 또는 필터라는 nXm 크기의 행렬로 높이X너비 크기의 이미지를 처음부터 끝까지 훑음
# nXm 크기의 겹쳐지는 부분의 이미지와 커널의 원소의 값을 곱해 모두 더한 값= 출력
# 이미지의 11시방향부터 5시방향으로 순차적으로 훑음
# 커널은 일반적으로 3x3, 5x5

# 합성곱 연산을 통해 나온 결과 = 특성맵(feature map)
# 이동범위 = 스트라이드(stride)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 패딩(Padding)
# 합성곱 연산시 특성맵은 입력보다 크기가 작아진다는 특징이 있음
# 이럴 경우 최종 특성맵은 초기 입력보다 매우 작아진 상태
# 이걸 방지해 동일한 크기를 유지하려면 패딩 사용

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 풀링(Pooling)
# 특성맵을 다운샘플링해 특성 맵의 크기를 줄이는 연산
# 일반적으로 최대풀링, 평균풀링 사용
# 학습해야 할 가중치가 없으며, 연산 후에 채널수가 변하지 않음
# 특성 맵의 가중치의 개수를 줄여줌