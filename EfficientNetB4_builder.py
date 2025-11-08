# EfficientNetB4_builder.py

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE 

# ImageNet 가중치의 Google Cloud Storage 경로
WEIGHTS_URL = (
    'https://storage.googleapis.com/keras-applications/'
    'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_no_top.h5'
)
WEIGHTS_HASH = '1e37036a166299d0c10c14ccf2d93e25' # EfficientNetB4의 가중치 해시값

def build_efficientnet_model(num_classes):
    """사전 훈련된 EfficientNetB4 모델을 기반으로 하는 전이 학습 모델을 생성하고 컴파일합니다."""
    
    # 1. ImageNet 가중치 없이 EfficientNetB4 모델 불러오기
    # 가중치 로딩 시 충돌을 피하기 위해 weights=None으로 생성합니다.
    base_model = EfficientNetB4(weights=None, 
                              include_top=False, 
                              input_shape=IMAGE_SIZE + (3,)) 

    # 2. 가중치 파일 다운로드 및 로드
    weights_path = tf.keras.utils.get_file(
        'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_no_top.h5',
        WEIGHTS_URL,
        cache_subdir='models',
        file_hash=WEIGHTS_HASH
    )
    
    try:
        # ⬇️ ImageNet 가중치 로드: skip_mismatch=True를 사용하여 1채널/3채널 충돌 레이어를 건너뜁니다.
        base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ ImageNet 가중치 로드 완료 (입력층 채널 충돌 무시).")
    except Exception as e:
        print(f"⚠️ ImageNet 가중치 로드 실패: {e}. 가중치 없이 새로 학습합니다.")
        
    # 3. 베이스 모델의 가중치를 동결(Freeze) (요청대로 유지)
    for layer in base_model.layers:
        layer.trainable = False

    # 4. 새로운 분류층을 추가하여 최종 모델 구성
    model = Sequential([
        base_model,                     
        GlobalAveragePooling2D(),       
        Dense(512, activation='relu'),  
        tf.keras.layers.Dropout(0.5),   
        Dense(NUM_CLASSES, activation='softmax') 
    ])

    # 5. 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # 테스트를 위한 임시 설정
    class MockConfig:
        IMAGE_SIZE = (299, 299)
        NUM_CLASSES = 10
        LEARNING_RATE = 0.0001
    
    model = build_efficientnet_model(MockConfig.NUM_CLASSES)
    model.summary()