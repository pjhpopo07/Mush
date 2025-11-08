# EfficientNetB4_builder.py

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file # 가중치 파일 다운로드를 위해 import
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE 

# ImageNet 가중치 경로 정의 (다운로드 경로)
WEIGHTS_URL = (
    'https://storage.googleapis.com/keras-applications/'
    'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_no_top.h5'
)
WEIGHTS_HASH = '1e37036a166299d0c10c14ccf2d93e25'

def build_efficientnet_model(num_classes):
    """사전 훈련된 EfficientNetB4 모델 기반의 전이 학습 모델을 생성하고 컴파일합니다."""
    
    # 1. Base Model 로드 (가중치 없이 시작)
    base_model = EfficientNetB4(
        weights=None, # ⬅️ None으로 설정하고 수동 로드 준비
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    )
    
    # 2. ImageNet 가중치 로드 (네트워크 문제 및 채널 불일치 우회)
    try:
        weights_path = tf.keras.utils.get_file(
            'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_no_top.h5',
            WEIGHTS_URL,
            cache_subdir='models',
            file_hash=WEIGHTS_HASH
        )
        # ⬇️ 핵심: skip_mismatch=True로 1채널/3채널 충돌 레이어를 건너뛰고 나머지 로드
        base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ ImageNet 가중치 로드 완료 (입력층 채널 충돌 무시).")
    
    except Exception as e:
        print(f"⚠️ ImageNet 가중치 로드 실패: {e}. 가중치 없이 새로 학습합니다.")

    # 3. 베이스 모델 동결 (Freeze) (요청대로 유지)
    base_model.trainable = False

    # 4. 분류기(Top Head) 추가 및 모델 구성
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5), 
        Dense(num_classes, activation='softmax')
    ])

    # 5. 모델 컴파일 (main에서 재컴파일되므로 기본값 유지)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model