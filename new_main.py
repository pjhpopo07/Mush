# main.py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

if not tf.executing_eagerly():
    tf.config.run_functions_eagerly(True)

import os
import datetime
from model_builder import build_xception_model
from data_processor import prepare_data_generators
from config import EPOCHS, LEARNING_RATE
import new_model_builder

# 모델 파일 저장 경로를 기존과 분리합니다.
MODEL_SAVE_PATH = "식용 이미지 추가한 CNN 모델.h5"

def scheduler(epoch, lr):
    # config.py에서 정의된 초기 학습률 (목표 LR)
    INITIAL_LR = LEARNING_RATE
    
    # --- 하이퍼파라미터 설정 ---
    WARMUP_EPOCHS = 3   # Ramp-up 구간: 3 에포크 동안 선형적으로 증가
    DECAY_STEP = 5      # Step-down 구간: 웜업 이후 5 에포크마다 감소
    DECAY_RATE = 0.1    # 감소율: 매 스텝마다 학습률이 10%로 줄어듦 (예: 0.0001 -> 0.00001)
    
    new_lr = 0
    
    # 1. Ramp Up (Warm-up) 구간
    if epoch < WARMUP_EPOCHS:
        # 0에서 시작하여 INITIAL_LR까지 선형적으로 증가
        new_lr = INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS # epoch+1로 0이 아닌 값부터 시작
        
    # 2. Step Down Decay (계단식 감소) 구간
    else:
        # WARMUP 이후부터 DECAY_STEP마다 학습률을 줄임
        steps = (epoch - WARMUP_EPOCHS) // DECAY_STEP
        new_lr = INITIAL_LR * (DECAY_RATE ** steps)
        
    print(f"Epoch {epoch + 1}/{EPOCHS}: Learning Rate is {new_lr:.8f}")
    return new_lr

def main():
    """프로젝트의 주요 흐름을 실행합니다: 데이터 준비, 모델 생성, 훈련."""
    
    # 1. 데이터 준비
    print("--- 1. 데이터 준비 중... ---")
    train_gen, val_gen, num_classes = prepare_data_generators()

    # 2. 모델 생성
    print("\n--- 2. Xception 모델 생성 중(12 클래스)... ---")
    initial_epoch = 0

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"✅ 기존 모델 '{MODEL_SAVE_PATH}' 로드 중...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        initial_epoch = 20 # 기존 학습 모델 에포크 수 20

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      run_eagerly=True)
        
    else:
        print("❌ 기존 모델이 없어 새로 생성합니다 (12 클래스).")
        model = build_xception_model(num_classes)
        initial_epoch = 0

    model.summary()

    # 3. 모델 훈련
    print(f"\n--- 3. 모델 훈련 시작 (Epochs: {initial_epoch + 1}부터 {EPOCHS}까지) ---")
    
    # 훈련 결과를 저장할 폴더 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", current_time) 

    # 로그 디렉터리 생성 (있으면 무시)
    os.makedirs(log_dir, exist_ok=True) 

     # TensorBoard 콜백 정의
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 

    # Learning Rate Scheduler 콜백 정의
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        callbacks=[tensorboard_callback]
    )

    # 4. 모델 저장
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- 4. 훈련 완료! 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다. ---")

    # 5. [추가] 훈련 결과 출력하기
    print("\n--- 5. 최종 훈련 결과 ---")

    # 훈련 정확도 (마지막 Epoch의 정확도)
    final_acc = history.history['accuracy'][-1] * 100
    # 검증 정확도 (모델의 실제 성능을 나타냄)
    final_val_acc = history.history['val_accuracy'][-1] * 100
    
    # 결과 출력
    print(f"최종 학습 정확도(Accuracy): {final_acc:.2f}%")
    print(f"최종 검증 정확도(Val Accuracy): {final_val_acc:.2f}%")

if __name__ == "__main__":
    main()