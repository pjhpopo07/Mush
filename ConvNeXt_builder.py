# ConvNeXt_builder.py

import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny # ConvNeXt 모델 사용 (예: Tiny)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE 

def build_convnext_model(num_classes):
    """사전 훈련된 ConvNeXt 모델을 기반으로 하는 전이 학습 모델을 생성하고 컴파일합니다."""
    
    # 1. Base Model 로드 (ConvNeXtTiny 사용)
    # ⚠️ 이 부분에서 네트워크 오류가 발생할 수 있습니다 (네트워크 환경 변경 필요).
    base_model = ConvNeXtTiny(
        weights='imagenet',
        include_top=False,
        input_shape=IMAGE_SIZE + (3,)
    ) 

    # 2. 베이스 모델의 가중치를 동결(Freeze) (요청대로 유지)
    for layer in base_model.layers:
        layer.trainable = False

    # 3. 새로운 분류층 추가
    model = Sequential([
        base_model,                     
        GlobalAveragePooling2D(),       
        Dense(512, activation='relu'),  
        Dropout(0.5),                   
        Dense(NUM_CLASSES, activation='softmax') 
    ])

    # 4. 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model