import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input 

# ==============================================================================
# ⚙️ 1. 설정 (필수 수정 영역)
# ==============================================================================
MODEL_PATH = '식용 이미지 추가한 CNN 모델.h5' 
IMAGE_SIZE = (299, 299) 
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', '맑은애주름버섯', 
               '붉은 뿔사슴버섯', '붉은점박이광대버섯', '영지버섯', '졸각버섯', '흰주름버섯'] 

SAFETY_MAPPING = {
    '개암버섯': '✅ 식용 버섯 (EDIBLE)', '노란개암버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯': '☠️ 맹독성 독버섯 (POISONOUS)', '마귀광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯': '🚨 맹독성 독버섯 (POISONOUS)', '붉은 뿔사슴버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '✅ 식용 버섯 (EDIBLE)', '영지버섯': '✅ 식용 버섯 (EDIBLE)', 
    '졸각버섯': '✅ 식용 버섯 (EDIBLE)', '흰주름버섯': '✅ 식용 버섯 (EDIBLE)',
}
# ==============================================================================

# --- 🎨 Custom CSS for Professional Look ---
def set_custom_style():
    st.markdown(
        """
        <style>
        /* 페이지 전체 폰트 및 배경 설정 */
        .main {
            background-color: #f0f2f6;
        }
        /* 메인 타이틀 */
        .stTitle {
            color: #4a5c6d;
            font-weight: 700;
        }
        /* 확신도 메트릭 강조 */
        [data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: bold;
            color: #008080; /* Teal color for emphasis */
        }
        /* 독버섯 경고 컨테이너 */
        .stAlert div[data-testid="stMarkdownContainer"] {
            font-size: 1.1rem;
            font-weight: bold;
            text-align: center;
        }
        /* 버튼 스타일 */
        div.stButton > button {
            background-color: #008080;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model_once(path):
    # (모델 로드 로직 유지)
    if not os.path.exists(path):
        st.error(f"🚨 오류: 모델 파일이 존재하지 않습니다: {os.path.basename(path)}")
        return None
    try:
        model = tf.keras.models.load_model(path)
        st.success(f"✅ Xception 모델 로드 성공: {os.path.basename(path)}")
        return model
    except Exception as e:
        st.error(f"🚨 치명적 오류: 모델 로드 실패 (호환성 문제): {e}")
        return None

def load_and_preprocess_image(image_file, target_size):
    # (이미지 전처리 로직 유지)
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def get_safety_status(mushroom_name):
    # (안전 상태 로직 유지)
    return SAFETY_MAPPING.get(mushroom_name, "⚠️ 정보 부족 - 추가 확인 필요")

# --- Streamlit UI 시작 ---
st.set_page_config(page_title="Xception 버섯 판독 시스템", layout="wide") # wide 레이아웃 사용
set_custom_style()

st.title("🍄 Xception 기반 버섯 안전 판독 시스템")
st.caption("AI 딥러닝 모델을 활용한 독/식용 버섯 판별 서비스")

# 1. 모델 로드
model = load_model_once(MODEL_PATH)

if model:
    # 2. 이미지 업로드 & 버튼 영역
    uploaded_file = st.file_uploader("판독할 버섯 이미지를 업로드하세요 (.jpg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        # 3. 레이아웃 분할 (이미지/결과를 나란히 배치)
        col1, col2 = st.columns([1, 1.5]) 
        
        with col1:
            st.markdown("### 🖼️ 업로드 이미지")
            st.image(uploaded_file, use_column_width=True)
            
            # 버튼을 이미지 아래에 배치
            if st.button("🍄 안전 판별 시작"):
                
                with st.spinner("모델이 예측 중입니다..."):
                    processed_image = load_and_preprocess_image(uploaded_file, IMAGE_SIZE)
                    predictions = model.predict(processed_image, verbose=0)
                    score = tf.nn.softmax(predictions[0])
                    
                    # 4. 결과 해석 및 TOP 3 계산
                    class_probabilities = list(zip(CLASS_NAMES, score.numpy()))
                    sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
                    top_n = 3
                    top_3_results = sorted_probabilities[:top_n]
                    
                    top_3_probs = [prob for name, prob in top_3_results]
                    sum_top_3_probs = sum(top_3_probs)
                    top_1_name = top_3_results[0][0]
                    
                    if sum_top_3_probs > 0:
                        top_1_confidence = (top_3_results[0][1] / sum_top_3_probs) * 100
                        safety_status = get_safety_status(top_1_name)
                    else:
                        top_1_confidence = 0.00
                        safety_status = "⚠️ 판독 불가능"

                # --- 5. 결과 출력 (col2에 출력) ---
                with col2:
                    st.markdown("### ✨ 최종 분석 결과")
                    
                    # 5-1. 안전 판별 (가장 눈에 띄게)
                    if 'POISONOUS' in safety_status or '☠️' in safety_status:
                        st.error(f"⚠️ 안전 판별: {safety_status}", icon="❌")
                    else:
                        st.success(f"✅ 안전 판별: {safety_status}", icon="🍄")
                        
                    st.markdown("---")
                    
                    # 5-2. 메인 지표 출력
                    st.metric(label="예측된 버섯 종류", value=top_1_name)
                    st.metric(label="모델 확신도 (재정규화)", value=f"{top_1_confidence:.2f}%")