import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 1. 모델 설정
IMAGE_SIZE = (299, 299)
NUM_CLASSES = 3  # 분류할 버섯 종류의 수에 맞게 변경하세요 (예: 3가지 종류)

# 2. 사전 훈련된 Xception 모델 불러오기
# weights='imagenet': ImageNet 데이터셋으로 학습된 가중치를 사용
# include_top=False: 모델의 원래 분류층(1000개 클래스)은 제거
base_model = Xception(weights='imagenet', 
                      include_top=False, 
                      input_shape=IMAGE_SIZE + (3,)) # (299, 299, 3)

# 3. 모델의 특징 추출 층을 동결 (Freeze)
# 기존에 학습된 능력은 그대로 보존하고 변경하지 않겠다는 의미입니다.
for layer in base_model.layers:
    layer.trainable = False

# 4. 새로운 분류층을 추가하는 모델 정의
model = Sequential([
    base_model,                               # 사전 훈련된 Xception 특징 추출기
    GlobalAveragePooling2D(),                 # 2차원 특징 맵을 1차원 벡터로 변환
    Dense(1024, activation='relu'),           # 추가적인 은닉층 (선택 사항)
    Dense(NUM_CLASSES, activation='softmax')  # 최종 분류층 (버섯 종류의 수에 맞게)
])

# 5. 모델 컴파일
# 여러분의 문제에 맞는 손실 함수(loss)와 최적화 도구(optimizer)를 설정합니다.
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # 다중 분류에 적합한 손실 함수
              metrics=['accuracy'])

# 모델 구조 요약 확인
model.summary()

# 이제 model.fit()을 사용하여 여러분의 데이터로 훈련을 시작하면 됩니다.