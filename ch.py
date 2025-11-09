import tensorflow as tf

# 1. 원본 모델 로드
original_model_path = 'ConvNext.keras'
new_h5_path = 'ConvNext.h5'

print(f"--- 원본 모델 로드 중: {original_model_path} ---")
try:
    model = tf.keras.models.load_model(original_model_path)
    
    # 2. 새로운 H5 포맷으로 저장
    print(f"--- 모델을 H5 포맷으로 변환 및 저장 중: {new_h5_path} ---")
    model.save(new_h5_path, save_format='h5')
    
    print("✅ 변환 성공! 이제 best_model_converted.h5 파일을 사용하세요.")

except Exception as e:
    print(f"❌ 오류 발생: 모델 로드 또는 저장에 실패했습니다. {e}")