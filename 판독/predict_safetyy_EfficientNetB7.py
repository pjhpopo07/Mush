import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ==============================================================================
# ⚙️ 1. 모델 파일 및 설정 (ConvNeXt에 맞게 수정)
# ==============================================================================
# 훈련 시 저장한 ConvNeXt 모델 파일 경로를 지정합니다.
MODEL_PATH = 'EfficientNetB7.keras' 
# ConvNeXt 모델이 훈련된 크기(예: 224x224, 299x299 등)에 맞춰야 합니다.
IMAGE_SIZE = (244, 244) 

# 판독할 새로운 이미지 경로 (!!! 실행 전에 이 경로를 반드시 수정하세요 !!!)
TEST_IMAGE_PATH = 'reading_images/images.jpg' 


# ==============================================================================
# 2. 사용자 설정: 클래스 이름 및 안전 정보 정의 (10개 클래스)
# ==============================================================================
# ⬇️ 주의: 이 리스트의 순서는 訓練時 데이터 폴더 순서와 EXACTLY 일치해야 합니다.
CLASS_NAMES = ['개암버섯',
               '노란개암버섯', 
               '독우산광대버섯', 
               '마귀광대버섯', 
               '맑은애주름버섯', 
               '붉은 뿔사슴버섯',
               '붉은점박이광대버섯',
               '영지버섯',
               '졸각버섯',
               '흰주름버섯',
               ] 

# 각 버섯 종류에 대한 안전 상태를 정의
SAFETY_MAPPING = {
    '개암버섯': '✅ 식용 버섯 (EDIBLE)', 
    '노란개암버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '마귀광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은 뿔사슴버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '✅ 식용 버섯 (EDIBLE)', 
    '영지버섯': '✅ 식용 버섯 (EDIBLE)', 
    '졸각버섯': '✅ 식용 버섯 (EDIBLE)', 
    '흰주름버섯': '✅ 식용 버섯 (EDIBLE)', 
}


def load_and_preprocess_image(image_path, target_size):
    """이미지를 로드하고 ConvNeXt 모델 입력에 맞게 전처리"""
    try:
        # 이미지 로드 및 전처리
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # ConvNeXt 모델 전처리 함수 사용
        img_preprocessed = tf.keras.applications.convnext.preprocess_input(img_array)
        
        return img_preprocessed
        
    except FileNotFoundError:
        print(f"오류: 이미지를 찾을 수 없습니다. 경로를 확인하세요: {image_path}")
        return None
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        return None

def get_safety_status(mushroom_name):
    """버섯 이름으로 독/식용 상태를 확인합니다."""
    return SAFETY_MAPPING.get(mushroom_name, "⚠️ 정보 부족 - 추가 확인 필요")

def predict_safety(model_path, image_path):
    """훈련된 ConvNeXt 모델을 사용하여 이미지를 예측하고 안전 여부를 판별합니다."""
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일이 존재하지 않습니다. 훈련된 모델('{model_path}')을 찾을 수 없습니다.")
        return

    # 1. 모델 로드
    print("--- 1. ConvNeXt 모델 로드 중... ---")
    model = tf.keras.models.load_model(model_path) 

    # ⬇️ [핵심 수정] 모델 로드 오류 우회를 위해 특징 추출기와 분류층 분리
    # 최종 분류층 (Dense)
    classifier_layer = model.layers[-1]
    
    # 분류층을 제외한 특징 추출기 (Head 이전)
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output # 분류층 이전 레이어의 출력을 특징으로 사용
    )
    
    # 2. 이미지 전처리
    print(f"--- 2. 이미지 '{os.path.basename(image_path)}' 전처리 중... ---")
    processed_image = load_and_preprocess_image(image_path, IMAGE_SIZE)
    if processed_image is None:
        return

    # 3. 예측 수행 (분리된 모델 사용)
    # 특징 추출기를 먼저 실행하여 1D 벡터(GlobalAveragePooling 이후)를 얻습니다.
    # ConvNeXt의 저장 방식에 따라 특징 추출기 출력이 이미 1D 벡터일 가능성이 높습니다.
    features = feature_extractor.predict(processed_image, verbose=0)
    
    # 4. 최종 예측: 분류층만 통과
    predictions = classifier_layer(features)
    
    # 5. 결과 해석 (확률값으로 변환)
    score = tf.nn.softmax(predictions[0]) 
    
    # TOP 3 재정규화 로직 시작
    class_probabilities = list(zip(CLASS_NAMES, score.numpy()))
    sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
    top_n = 3
    top_3_results = sorted_probabilities[:top_n]
    
    top_3_probs = [prob for name, prob in top_3_results]
    sum_top_3_probs = sum(top_3_probs)
    
    top_1_name = top_3_results[0][0]
    top_1_prob_original = top_3_results[0][1]
    
    # 6. 재정규화 및 결과 설정
    if sum_top_3_probs > 0:
        top_1_confidence = (top_1_prob_original / sum_top_3_probs) * 100
        predicted_name = top_1_name
        safety_status = get_safety_status(predicted_name)
    else:
        top_1_confidence = 0.00
        predicted_name = "판독 실패"
        safety_status = "⚠️ 정보 부족 - 추가 확인 필요"

    # 7. 최종 결과 출력 (TOP 3 기준)
    print("\n--- 3. 판독 및 안전 결과 ---")
    print(f"🍄 예측된 버섯 종류: {predicted_name}") 
    print(f"✨ 모델 확신도: {top_1_confidence:.2f}%") 
    print("-" * 35)
    print(f"🚨 독/식용 최종 판별: {safety_status}")
    print("-" * 35)
    
    # 8. 재정규화된 TOP 3 출력
    print(f"\n[✨ TOP {top_n} 클래스 재정규화 확신도 결과 (총합 100%) ✨]")
        
    if sum_top_3_probs > 0:
        for i, (name, prob) in enumerate(top_3_results):
            renormalized_prob = (prob / sum_top_3_probs) * 100
            print(f"  {i+1}. {name}: {renormalized_prob:.2f}%")
    else:
        print("  합계 확률이 0이어서 재정규화할 수 없습니다.")
    
if __name__ == '__main__':
    # Pillow (PIL) 라이브러리가 필요합니다.
    predict_safety(MODEL_PATH, TEST_IMAGE_PATH)