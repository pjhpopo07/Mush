# ConvNeXt_main.py

import tensorflow as tf
import os
import datetime
# ConvNeXt 빌더 파일과 함수 이름으로 import
from ConvNext_builder import build_convnext_model 
import config as cfg # config 파일 import
from data_processor import prepare_data_generators
from tensorflow.keras.callbacks import LearningRateScheduler

# 모델 파일 저장 경로 정의
MODEL_SAVE_PATH = "new_convnext_model_10_classes.keras"

def scheduler(epoch, lr):
    """Warm-up과 Step Decay가 적용된 학습률 스케줄러"""
    INITIAL_LR = cfg.LEARNING_RATE
    WARMUP_EPOCHS = 3   
    DECAY_STEP = 5      
    DECAY_RATE = 0.1    
    
    new_lr = 0
    
    if epoch < WARMUP_EPOCHS:
        # Warm-up 구간: 선형적으로 증가
        new_lr = INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS 
    else:
        # Step Decay 구간
        steps = (epoch - WARMUP_EPOCHS) // DECAY_STEP
        new_lr = INITIAL_LR * (DECAY_RATE ** steps)
        
    print(f"Epoch {epoch + 1}/{cfg.EPOCHS}: Learning Rate is {new_lr:.8f}")
    return new_lr

def main():
    """ConvNeXt 모델 학습의 주요 흐름을 실행합니다."""
    
    # 1. 데이터 준비
    print("--- 1. 데이터 준비 중... ---")
    # data_processor.py에 prepare_data_generators 함수가 있다고 가정합니다.
    train_gen, val_gen, num_classes = prepare_data_generators()

    # 2. 모델 생성
    print(f"\n--- 2. ConvNeXt 모델 생성 중 ({num_classes} 클래스)... ---")
    
    # Eager Execution 강제 활성화 (TensorFlow 오류 방지용)
    if not tf.executing_eagerly():
        tf.config.run_functions_eagerly(True)

    # 모델 생성
    model = build_convnext_model(num_classes) 
    initial_epoch = 0

    model.summary()

    # 3. 모델 훈련
    print(f"\n--- 3. 모델 훈련 시작 (Epochs: 1부터 {cfg.EPOCHS}까지) ---")
    
    # ⬇️ 🚨 [NameError 해결] log_dir 변수를 먼저 정의합니다. 🚨
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit_convnext", current_time) 
    os.makedirs(log_dir, exist_ok=True) 
    
    # ⬇️ 🚨 [NameError 해결] 콜백 변수들을 model.fit() 전에 정의합니다. 🚨
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 
    lr_scheduler_callback = LearningRateScheduler(scheduler)

    # 훈련 실행
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