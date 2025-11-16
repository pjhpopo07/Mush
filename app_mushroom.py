import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.convnext import preprocess_input # ConvNeXt 전처리 함수

# ==============================================================================
# ⚙️ 1. 설정 (이 부분을 수정하세요)
# ==============================================================================
# ⚠️ 두 모델 파일 경로를 모두 지정합니다. (코드가 둘 다 로드 시도)
MODEL_PATH_KERAS = 'ConvNext.keras' 
MODEL_PATH_H5 = '식용 이미지 추가한 CNN 모델.h5' # H5 변환 파일명 (만약 있다면)
IMAGE_SIZE = (299, 299) 
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', '맑은애주름버섯', '붉은 뿔사슴버섯', '붉은 점박이광대버섯', '영지버섯', '졸각버섯', '흰주름버섯'] # ⬅️ 2개 클래스 또는 10개 클래스에 맞게 수정

# SAFETY_MAPPING (클래스 이름과 일치하도록 수정)
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
# ==============================================================================

@st.cache_resource
def load_model_from_files():
    """두 파일 경로를 시도하여 모델을 로드합니다."""
    for path in [MODEL_PATH_KERAS, MODEL_PATH_H5]:
        if os.path.exists(path):
            try:
                # Keras의 load_model은 .keras와 .h5 포맷 모두 호환하여 로드합니다.
                model = tf.keras.models.load_model(path)
                st.success(f"✅ 모델 로드 성공: {os.path.basename(path)}")
                return model
            except Exception as e:
                st.error(f"🚨 오류: {os.path.basename(path)} 로드 실패. 저장 형식 불일치 문제일 수 있습니다. {e}")
    st.error("🚨 치명적 오류: 유효한 모델 파일(.keras 또는 .h5)을 찾을 수 없습니다.")
    return None

def load_and_preprocess_image(image_file, target_size):
    """업로드된 이미지 파일을 로드하고 xeception 전처리를 수행합니다."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # xception 모델 전처리 함수 사용
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

# --- Streamlit UI 시작 ---
st.set_page_config(page_title="버섯 이미지 판독 시스템", layout="centered")

st.title("🍄 xception 기반 버섯 안전 판독 시스템")
st.markdown("---")

# 1. 모델 로드
model = load_model_from_files()

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
                
                # 4. 결과 해석 (Top 1 및 안전 등급)
                class_names = [name for name in SAFETY_MAPPING.keys()] # 클래스 이름을 SAFETY_MAPPING에서 가져옴
                
                # TOP 1 이름과 확률 추출
                predicted_index = np.argmax(score)
                predicted_name = class_names[predicted_index]
                confidence = np.max(score) * 100
                
                safety_status = SAFETY_MAPPING.get(predicted_name, "⚠️ 정보 부족")

                # 5. 결과 출력
                st.subheader("✅ 최종 판독 결과")
                
                st.metric(label="예측된 종류 (Top 1)", value=predicted_name)
                st.metric(label="모델 확신도", value=f"{confidence:.2f}%")
                
                if 'POISONOUS' in safety_status:
                    st.error(f"🚨 독/식용 판별: {safety_status}")
                else:
                    st.success(f"✅ 독/식용 판별: {safety_status}")
                    
                st.markdown("---")
                
                # (선택적) 모든 클래스 확신도 출력
                st.markdown("**🔍 상세 확신도 (모든 클래스)**")
                for name, prob in zip(class_names, score):
                    st.write(f"- {name}: {prob*100:.2f}%")