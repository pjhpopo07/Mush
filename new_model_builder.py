# model_builder.py

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE # config 파일에서 설정값 불러오기
from tensorflow.keras import backend as K

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Categorical Focal Loss를 Keras/TensorFlow에서 직접 구현합니다.
    논문 권장값: gamma=2.0, alpha=0.25
    """
    
    def focal_loss(y_true, y_pred):
        # 1. 예측 확률값 클리핑 (log(0) 방지)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # 2. 교차 엔트로피 계산 (Cross-Entropy: CE)
        cross_entropy = -y_true * K.log(y_pred) 
        
        # 3. 정답에 대한 예측 확률 p_t 추출
        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        
        # 4. 변조 계수 계산 (Modulating Factor): (1 - p_t)^gamma
        modulating_factor = K.pow(1. - p_t, gamma)
        
        # 5. Alpha (가중치 계수) 적용
        alpha_factor = alpha * y_true + (1. - alpha) * (1. - y_true)
        
        # 6. Focal Loss 최종 계산
        focal_loss = alpha_factor * modulating_factor * cross_entropy

        # 7. 배치 전체의 평균 손실 반환
        return K.sum(focal_loss, axis=-1)

    return focal_loss

def build_xception_model(num_classes):
    """사전 훈련된 Xception 모델을 기반으로 하는 전이 학습 모델을 생성하고 컴파일합니다."""
    
    # 1. 사전 훈련된 Xception 모델 불러오기 (특징 추출기)
    # include_top=False로 설정하여 원래의 분류층은 제거합니다.
    base_model = Xception(weights='imagenet', 
                          include_top=False, 
                          input_shape=IMAGE_SIZE + (3,)) # (299, 299, 3)

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
    model = build_xception_model()
    model.summary()