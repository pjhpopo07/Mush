import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.convnext import preprocess_input 

# ==============================================================================
# ⚙️ 1. 모델 및 경로 설정
# ==============================================================================
# 훈련 시 저장한 ConvNeXt 모델 파일 경로를 지정합니다.
MODEL_PATH = 'ConvNeXt.keras' 
IMAGE_SIZE = (244, 244) 
TEST_IMAGE_PATH = 'reading_images/test_mushroom.jpg' 

# 2. 사용자 설정 (클래스 이름과 안전 정보는 이전 답변에서 정의된 것을 사용)
# ⚠️ 이 리스트의 순서는 訓練時 데이터 폴더 순서와 EXACTLY 일치해야 합니다.
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', 
               '맑은애주름버섯', '붉은 뿔사슴버섯', '붉은점박이광대버섯',
               '영지버섯', '졸각버섯', '흰주름버섯'] 
SAFETY_MAPPING = {
    '개암버섯': '✅ 식용 버섯 (EDIBLE)', '노란개암버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯': '🚨 맹독성 독버섯 (POISONOUS)', '마귀광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯': '🚨 맹독성 독버섯 (POISONOUS)', '붉은 뿔사슴버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '✅ 식용 버섯 (EDIBLE)', '영지버섯': '✅ 식용 버섯 (EDIBLE)', 
    '졸각버섯': '✅ 식용 버섯 (EDIBLE)', '흰주름버섯': '✅ 식용 버섯 (EDIBLE)',
}
# -----------------------------------------------------------------------------

def get_safety_status(mushroom_name):
    return SAFETY_MAPPING.get(mushroom_name, "⚠️ 정보 부족 - 추가 확인 필요")

def load_class_names(train_data_dir):
    # 이 부분은 현재 모델이 훈련 데이터 폴더 없이 실행되므로, CLASS_NAMES를 직접 반환합니다.
    return CLASS_NAMES

def get_safety_mapping(class_names):
    """클래스 이름을 기반으로 독/식용 매핑을 정의합니다."""
    safety_map = {}
    # ... (10개 클래스에 대한 독/식용 매핑 로직) ...
    return safety_map

def load_and_preprocess_image(image_path, target_size):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed
    except Exception as e:
        print(f"🚨 이미지 처리 중 오류 발생: {e}")
        return None


def predict_safety():
    """모델 로드, 전처리, 예측 및 결과 출력을 수행합니다."""
    
    # 1. 설정 로드
    class_names = load_class_names("") # 고정된 CLASS_NAMES 사용
    safety_map = get_safety_mapping(class_names)
    
    # 2. 모델 로드
    print("--- 1. ConvNeXt 모델 로드 중... ---")
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 오류: 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH) 

    # ⬇️ [최종 오류 우회] 특징 추출기와 분류층 분리 및 연결 복구
    try:
        # 최종 분류층 (Dense layer) 추출
        classifier_layer = model.layers[-1]
        
        # 특징 추출기: 분류층 이전 레이어의 출력을 얻어옵니다.
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
    except Exception as e:
        print(f"🚨 오류: 모델 구조 분리 실패. 저장된 모델이 호환되지 않습니다: {e}")
        return
    
    # 3. 이미지 전처리
    processed_image = load_and_preprocess_image(TEST_IMAGE_PATH, TARGET_SIZE)
    if processed_image is None: return

    # 4. 예측 수행 (분리된 모델 사용)
    features = feature_extractor.predict(processed_image, verbose=0)
    
    # ⬇️ [Tensor Shape 문제 해결] 4D 텐서 (None, 7, 7, 768)를 1D 벡터로 강제 변환
    if len(features.shape) > 2:
         print("⚠️ 4차원 특징 텐서 감지. GlobalAveragePooling2D로 강제 압축 중...")
         pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)
    else:
         pooled_features = features

    # 5. 최종 예측: 분류층만 통과
    predictions = classifier_layer(pooled_features) 
    score = tf.nn.softmax(predictions[0]) 
    
    # 6. 결과 출력 (TOP 3 재정규화 로직 유지)
    class_probabilities = list(zip(class_names, score.numpy()))
    sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
    top_n = 3
    top_3_results = sorted_probabilities[:top_n]
    
    top_3_probs = [prob for name, prob in top_3_results]
    sum_top_3_probs = sum(top_3_probs)
    top_1_name = top_3_results[0][0]

    if sum_top_3_probs > 0:
        top_1_confidence = (top_3_results[0][1] / sum_top_3_probs) * 100
        safety_status = get_safety_status(top_1_name, safety_map)
    else:
        top_1_confidence = 0.00
        top_1_name = "판독 실패"
        safety_status = "⚠️ 정보 부족 - 추가 확인 필요"

    print("\n--- 3. 판독 및 안전 결과 ---")
    print(f"🍄 예측된 버섯 종류: {top_1_name}") 
    print(f"✨ 모델 확신도: {top_1_confidence:.2f}%") 
    print("-" * 35)
    print(f"🚨 독/식용 최종 판별: {safety_status}")
    print("-" * 35)
    
    print(f"\n[✨ TOP {top_n} 클래스 재정규화 확신도 결과 (총합 100%) ✨]")
    
    if sum_top_3_probs > 0:
        for i, (name, prob) in enumerate(top_3_results):
            renormalized_prob = (prob / sum_top_3_probs) * 100
            print(f"  {i+1}. {name}: {renormalized_prob:.2f}%")
    else:
        print("  합계 확률이 0이어서 재정규화할 수 없습니다.")
    
if __name__ == '__main__':
    predict_safety()