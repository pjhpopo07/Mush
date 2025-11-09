import tensorflow as tf
import numpy as np
import os
from PIL import Image


# 1. 모델 파일 및 설정 (Xception에 맞게 수정)

# 훈련 시 저장한 Xception 모델 파일 경로로 수정하세요.
MODEL_PATH = 'focal_xception_model.h5' 
# Xception 모델의 표준 입력 크기는 299x299입니다.
IMAGE_SIZE = (299, 299) 

# 판독할 새로운 이미지 경로 (!!! 실행 전에 이 경로를 반드시 수정하세요 !!!)
TEST_IMAGE_PATH = 'reading_images/dog_usan.jpg' 


# 2. 사용자 설정: 클래스 이름 및 안전 정보 정의

# 주의: 이 리스트의 순서는 훈련 시 데이터 프로세서가 읽은 5개 폴더의 순서와 EXACTLY 일치해야 합니다.
# 실제 5개 클래스 이름으로 수정해야 합니다.
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
    '노란개암버섯': ' 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯':    '맹독성 독버섯 (POISONOUS)',
    '마귀광대버섯':   '맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯':   '맹독성 독버섯 (POISONOUS)',
    '붉은 뿔사슴버섯':   '맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '식용 버섯 (EDIBLE)', 
    '영지버섯': '식용 버섯 (EDIBLE)', 
    '졸각버섯': '식용 버섯 (EDIBLE)', 
    '개암버섯': '식용 버섯 (EDIBLE)', 
    '흰주름버섯': '식용 버섯 (EDIBLE)', 
}


def load_and_preprocess_image(image_path, target_size):
    """이미지를 로드하고 Xception 모델 입력에 맞게 전처리"""
    try:
        # 1. 이미지 로드 및 크기 조정
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        # 2. NumPy 배열로 변환
        img_array = tf.keras.utils.img_to_array(img)
        # 3. 배치 차원 추가 
        img_array = np.expand_dims(img_array, axis=0) 
        
        # !!! 핵심 변경 사항: Xception 모델 전처리 함수 사용 !!!
        # 이 함수는 픽셀 값을 -1과 1 사이로 변환하여 Xception의 훈련 방식과 일치시킵니다.
        img_preprocessed = tf.keras.applications.xception.preprocess_input(img_array)
        
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
    """훈련된 Xception 모델을 사용하여 이미지를 예측하고 안전 여부를 판별합니다."""
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일이 존재하지 않습니다. 훈련된 모델('{model_path}')을 찾을 수 없습니다.")
        return

    # 1. 모델 로드
    print("--- 1. Xception 모델 로드 중... ---")
    model = tf.keras.models.load_model(model_path)

    # 2. 이미지 전처리
    print(f"--- 2. 이미지 '{os.path.basename(image_path)}' 전처리 중... ---")
    processed_image = load_and_preprocess_image(image_path, IMAGE_SIZE)
    if processed_image is None:
        return

    # 3. 예측 수행
    predictions = model.predict(processed_image)
    
    # 4. 결과 해석 (확률값으로 변환)
    score = tf.nn.softmax(predictions[0]) 
  
    # 1. 클래스 이름과 확률을 쌍으로 묶습니다.
    # score.numpy()를 사용하여 TensorFlow 텐서를 NumPy 배열로 변환합니다.
    class_probabilities = list(zip(CLASS_NAMES, score.numpy()))
    
    # 2. 확률을 기준으로 내림차순 정렬합니다.
    sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
    top_n = 3
    top_3_results = sorted_probabilities[:top_n]
    
    # 3. 상위 3개 클래스만 선택합니다.
    top_3_probs = [prob for name, prob in top_3_results]
    sum_top_3_probs = sum(top_3_probs)
    
    # 4. 재정규화된 결과를 출력합니다.
    if sum_top_3_probs > 0:
        top_1_name = top_3_results[0][0]
        top_1_prob_original = top_3_results[0][1]

        top_1_confidence = (top_1_prob_original / sum_top_3_probs) * 100

        predicted_name = top_1_name
        safety_status = get_safety_status(predicted_name)

    else:
        # 합계가 0인 예외 상황 처리
        top_1_confidence = 0.00
        predicted_name = "판독 실패"
        safety_status = "⚠️ 정보 부족 - 추가 확인 필요"

        # 5. 최종 결과 출력 (TOP 3 기준)
    print("\n--- 3. 판독 및 안전 결과 ---")
    print(f"🍄 예측된 버섯 종류: {predicted_name}")
    print(f"✨ 모델 확신도: {top_1_confidence:.2f}%") # 재정규화된 TOP 1 확률 사용
    print("-" * 35)
    print(f"🚨 독/식용 최종 판별: {safety_status}")
    print("-" * 35)
    
    # 6. 재정규화된 TOP 3 출력
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