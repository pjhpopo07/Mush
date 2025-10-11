# main.py

import tensorflow as tf
import os
import datetime
from model_builder import build_xception_model
from data_processor import prepare_data_generators
from config import EPOCHS

def main():
    """프로젝트의 주요 흐름을 실행합니다: 데이터 준비, 모델 생성, 훈련."""
    
    # 1. 데이터 준비
    print("--- 1. 데이터 준비 중... ---")
    train_gen, val_gen = prepare_data_generators()

    # 2. 모델 생성
    print("\n--- 2. Xception 모델 생성 중... ---")
    model = build_xception_model()
    model.summary()

    # 3. 모델 훈련
    print(f"\n--- 3. 모델 훈련 시작 (Epochs: {EPOCHS}) ---")
    
    # 훈련 결과를 저장할 폴더 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "fit", current_time) 

    # 로그 디렉터리 생성 (있으면 무시)
    os.makedirs(log_dir, exist_ok=True) 

     # TensorBoard 콜백 정의
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[tensorboard_callback]
    )

    # 4. 모델 저장
    model_save_path = "xception_mushroom_classifier.h5"
    model.save(model_save_path)
    print(f"\n--- 4. 훈련 완료! 모델이 '{model_save_path}'에 저장되었습니다. ---")

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