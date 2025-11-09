# EfficientNetB4_main.py

import tensorflow as tf
import os
import datetime
from EfficientNetB4.EfficientNetB4_builder import build_efficientnet_model 
import config as cfg 
from data_processor import prepare_data_generators
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam # 재컴파일을 위해 Adam 옵티마이저 import

# 모델 파일 저장 경로를 정의합니다.
MODEL_SAVE_PATH = "EfficientNetB4_10_classes.h5"

def scheduler(epoch, lr):
    # config_efficientnet.py에서 정의된 초기 학습률 사용
    INITIAL_LR = cfg.LEARNING_RATE
    
    # --- Learning Rate Scheduler 로직 (유지) ---
    WARMUP_EPOCHS = 3   
    DECAY_STEP = 5      
    DECAY_RATE = 0.1    
    
    new_lr = 0
    
    if epoch < WARMUP_EPOCHS:
        new_lr = INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS
    else:
        steps = (epoch - WARMUP_EPOCHS) // DECAY_STEP
        new_lr = INITIAL_LR * (DECAY_RATE ** steps)
        
    print(f"Epoch {epoch + 1}/{cfg.EPOCHS}: Learning Rate is {new_lr:.8f}")
    return new_lr

def main():
    """EfficientNet 모델 학습의 주요 흐름을 실행합니다."""
    
    # 1. 데이터 준비
    print("--- 1. 데이터 준비 중... ---")
    train_gen, val_gen, num_classes = prepare_data_generators()

    # 2. 모델 로드 또는 생성 (10개 클래스)
    print(f"\n--- 2. EfficientNet 모델 로드 또는 생성 중 ({num_classes} 클래스)... ---")
    
    # 📌 Eager Execution 강제 활성화 (오류 방지)
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    initial_epoch = 0 
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"✅ 기존 모델 '{MODEL_SAVE_PATH}' 로드 중...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        # 이전 학습 에포크 수 수동 지정 (필요 시 수정, 여기서는 20으로 가정)
        initial_epoch = 20 
        
        # ⚠️ [핵심 수정] 로드 후 재컴파일 (ValueError 해결)
        model.compile(optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      run_eagerly=True) # Eager 실행 강제
        print("✅ 모델 재컴파일 완료.")

    else:
        print("❌ 기존 모델이 없어 새로 EfficientNet을 생성합니다.")
        model = build_efficientnet_model(num_classes)
        initial_epoch = 0

    model.summary()

    # 3. 모델 훈련
    print(f"\n--- 3. 모델 훈련 시작 (Epochs: {initial_epoch + 1}부터 {cfg.EPOCHS}까지) ---")
    
    # 훈련 결과를 저장할 폴더 생성 (로그 관리)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit_efficientnet", current_time) 
    os.makedirs(log_dir, exist_ok=True) 

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 
    lr_scheduler_callback = LearningRateScheduler(scheduler)

    history = model.fit(
        train_gen,
        epochs=cfg.EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        callbacks=[tensorboard_callback, lr_scheduler_callback]
    )

    # 4. 모델 저장
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- 4. 훈련 완료! 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다. ---")

if __name__ == "__main__":
    main()