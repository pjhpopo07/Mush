import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
# Xception 모델 전처리 함수를 사용합니다.
from tensorflow.keras.applications.xception import preprocess_input 

# ==============================================================================
# ⚙️ 1. 설정 (필수 수정 영역)
# ==============================================================================
# 훈련 시 저장한 Xception 모델 파일 경로를 지정합니다.
MODEL_PATH = '식용 이미지 추가한 CNN 모델.h5' 
IMAGE_SIZE = (299, 299) # Xception 표준 크기 유지

# ⚠️ 이 리스트의 순서는 訓練時 데이터 폴더 순서와 EXACTLY 일치해야 합니다.
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', '맑은애주름버섯', 
               '붉은 뿔사슴버섯', '붉은점박이광대버섯', '영지버섯', '졸각버섯', '흰주름버섯'] 

# SAFETY_MAPPING (10개 클래스의 독/식용 안전 정보)
SAFETY_MAPPING = {
    '개암버섯': '✅ 식용 버섯 (EDIBLE)', '노란개암버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯': '🚨 맹독성 독버섯 (POISONOUS)', '마귀광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯': '🚨 맹독성 독버섯 (POISONOUS)', '붉은 뿔사슴버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '✅ 식용 버섯 (EDIBLE)', '영지버섯': '✅ 식용 버섯 (EDIBLE)', 
    '졸각버섯': '✅ 식용 버섯 (EDIBLE)', '흰주름버섯': '✅ 식용 버섯 (EDIBLE)',
}
# ==============================================================================

@st.cache_resource
def load_model_once(path):
    """모델 파일을 로드합니다."""
    if not os.path.exists(path):
        st.error(f"🚨 오류: 모델 파일이 존재하지 않습니다: {os.path.basename(path)}")
        return None
    try:
        model = tf.keras.models.load_model(path)
        st.success(f"✅ Xception 모델 로드 성공: {os.path.basename(path)}")
        return model
    except Exception as e:
        st.error(f"🚨 치명적 오류: 모델 로드 실패 (저장 형식 문제일 수 있습니다). 오류: {e}")
        return None

def load_and_preprocess_image(image_file, target_size):
    """업로드된 이미지 파일을 로드하고 Xception 전처리를 수행합니다."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # ⬇️ Xception 모델 전처리 함수 사용
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def get_safety_status(mushroom_name):
    """버섯 이름으로 독/식용 상태를 확인합니다."""
    return SAFETY_MAPPING.get(mushroom_name, "⚠️ 정보 부족 - 추가 확인 필요")

# --- Streamlit UI 시작 ---
st.set_page_config(page_title="Xception 버섯 판독 시스템", layout="centered")

st.title("🍄 Xception 기반 버섯 안전 판독 시스템")
st.markdown("---")

# 1. 모델 로드
model = load_model_once(MODEL_PATH)

if model:
    # 2. 이미지 업로드
    uploaded_file = st.file_uploader("판독할 버섯 이미지를 업로드하세요 (.jpg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드된 버섯 이미지", use_column_width=True)
        st.markdown("---")
        
        # 3. 예측 실행
        if st.button("🍄 안전 판별 시작"):
            with st.spinner("모델이 예측 중입니다..."):
                # 전처리 수행
                processed_image = load_and_preprocess_image(uploaded_file, IMAGE_SIZE)
                
                # 예측 수행
                predictions = model.predict(processed_image, verbose=0)
                score = tf.nn.softmax(predictions[0])
                
                # 4. 결과 해석 및 TOP 3 계산
                class_probabilities = list(zip(CLASS_NAMES, score.numpy()))
                sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
                top_n = 3
                top_3_results = sorted_probabilities[:top_n]
                
                top_3_probs = [prob for name, prob in top_3_results]
                sum_top_3_probs = sum(top_3_probs)
                
                # Top 1 결과
                top_1_name = top_3_results[0][0]
                top_1_prob_original = top_3_results[0][1]
                
                # 재정규화 및 최종 판별
                if sum_top_3_probs > 0:
                    top_1_confidence = (top_1_prob_original / sum_top_3_probs) * 100
                    safety_status = get_safety_status(top_1_name)
                else:
                    top_1_confidence = 0.00
                    safety_status = "⚠️ 판독 불가능"

                # 5. UI 출력
                st.subheader("✅ 최종 판독 결과")
                
                st.metric(label="🍄 예측된 버섯 종류", value=top_1_name)
                st.metric(label="✨ 모델 확신도 (재정규화)", value=f"{top_1_confidence:.2f}%")
                
                if 'POISONOUS' in safety_status:
                    st.error(f"🚨 최종 안전 판별: {safety_status}")
                else:
                    st.success(f"✅ 최종 안전 판별: {safety_status}")
                    
                st.markdown("---")
                
                # ⬇️ TOP 3 결과를 단일 테이블로 출력
                st.subheader("📊 TOP 3 예측 상세 분석")
                
                data_for_table = []
                for idx, (name, prob) in enumerate(top_3_results):
                    renormalized_prob = (prob / sum_top_3_probs) * 100 if sum_top_3_probs > 0 else 0.0
                    data_for_table.append([f"{idx+1}.", name, f"{renormalized_prob:.2f}%"])
                
                st.table(data_for_table)