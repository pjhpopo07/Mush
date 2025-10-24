# model_builder.py

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE, NUM_CHANNELS # config 파일에서 설정값 불러오기

def build_efficientnetb3_model(num_classes):
    """사전 훈련된 efficientnetb3_model 모델을 기반으로 하는 전이 학습 모델을 생성하고 컴파일합니다."""
    
    input_shape_with_channels = IMAGE_SIZE + (NUM_CHANNELS,) # (300, 300, 3)
    # 1. 사전 훈련된 Xception 모델 불러오기 (특징 추출기)
    # include_top=False로 설정하여 원래의 분류층은 제거합니다.
    base_model = EfficientNetB3(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape_with_channels)

    # 2. 베이스 모델의 가중치를 동결(Freeze)
    # 기존에 학습된 특징 추출 능력을 보호합니다.
    for layer in base_model.layers:
        layer.trainable = False

    # 3. 새로운 분류층을 추가하여 최종 모델 구성
    model = Sequential([
        base_model,                     # Xception 특징 추출기
        GlobalAveragePooling2D(),       # 2D 특징 맵을 1D 벡터로 압축
        Dense(512, activation='relu'),  # 추가 은닉층
        tf.keras.layers.Dropout(0.5),   # Dropout(0.5) 계층 추가
        Dense(NUM_CLASSES, activation='softmax') # 최종 분류층
    ])

    # 4. 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', # 다중 분류 손실 함수
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # 모델 구조를 간단히 확인하기 위한 코드
    from config import NUM_CLASSES

    model = build_efficientnetb3_model(NUM_CLASSES)
    model.summary()