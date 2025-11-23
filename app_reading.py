import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pandas as pd
import base64 # Base64 인코딩을 위해 필요
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# ==============================================================================
# ⚠️ 1. 모델 파일 및 설정 (경로 확인 필수)
# ==============================================================================
# ConvNeXt 모델 설정
CONVNEXT_MODEL_PATH = 'ConvNeXt.keras'  # ConvNeXt .keras 파일 경로
CONVNEXT_IMAGE_SIZE = (244, 244)             

# Xception 모델 설정
XCEPTION_MODEL_PATH = '식용 이미지 추가한 CNN 모델.h5' # Xception .h5 파일 경로
XCEPTION_IMAGE_SIZE = (299, 299)             

# [필수] 訓練時 데이터 폴더 순서와 EXACTLY 일치해야 합니다. (10개 클래스 가정)
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', '맑은애주름버섯', 
               '붉은 뿔사슴버섯', '붉은점박이광대버섯', '영지버섯', '졸각버섯', '흰주름버섯'] 

# SAFETY_MAPPING (10개 클래스의 독/식용 안전 정보)
SAFETY_MAPPING = {
    '개암버섯': '✅ 식용 버섯 (EDIBLE)', '노란개암버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '독우산광대버섯': '☠️ 맹독성 독버섯 (POISONOUS)', '마귀광대버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '맑은애주름버섯': '🚨 맹독성 독버섯 (POISONOUS)', '붉은 뿔사슴버섯': '🚨 맹독성 독버섯 (POISONOUS)',
    '붉은점박이광대버섯': '✅ 식용 버섯 (EDIBLE)', '영지버섯': '✅ 식용 버섯 (EDIBLE)', 
    '졸각버섯': '✅ 식용 버섯 (EDIBLE)', '흰주름버섯': '✅ 식용 버섯 (EDIBLE)',
}
# ==============================================================================

# 🖼️ 배경 이미지 설정
# ------------------------------------------------------------------------------------------------
BACKGROUND_IMAGE_FILENAME = 'forest-7406241_1280.jpg' 

def get_base64_of_image(image_path):
    """이미지 파일을 읽어 Base64 문자열로 변환합니다."""
    if not os.path.exists(image_path):
        return None
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # ⬇️ [핵심 수정] 줄바꿈 및 공백 문자를 제거하여 CSS 구문 오류를 방지합니다.
        return encoded_string.replace('\n', '').strip() 
        
    except Exception as e:
        print(f"🚨 배경 이미지 파일 로드 오류: {e}")
        return None

def set_custom_style(background_image_base64):
    """Streamlit 앱의 제목 영역에만 배경 이미지 CSS를 적용합니다."""
    st.markdown(
        f"""
        <style>
        /* 1. [최종 수정] HTML BODY 자체에 배경을 설정하여 이미지 누락 방지 */
        /* 이 방법은 가장 포괄적인 배경 설정을 보장합니다. */
        body {{
            background-image: url("data:image/jpeg;base64,{background_image_base64}");
            background-size: cover; 
            background-position: center; 
            background-repeat: no-repeat; 
            background-attachment: fixed;
            height: 150px;
            padding: 0 !important;

        }}
        /* 2. Streamlit의 기본 배경색을 제거하여 이미지 보이도록 함 */
        .stApp, .block-container, [data-testid="stHeader"] ~div{{
            background-color: transparent !important; 
        }}
        
        /* 3. 메인 콘텐츠 영역의 배경을 반투명하게 설정 */
        .block-container {{ 
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        /* 4. 제목 텍스트 스타일 (유지) */
        h1 {{
            color: #ffffff; 
            text-shadow: 2px 2px 5px #000000;
            text-align: center;
            font-size: 3.5em; 
            padding-top: 0px; 
            padding-bottom: 0px; 
            line-height: 150px;
            margin-bottom: 0;
        }}
        /* 기타 스타일 유지 */
        </style>
        """,
        unsafe_allow_html=True
    )
    
# ------------------------------------------------------------------------------------------------


# --- 함수 정의 ---

@st.cache_resource
def load_all_models():
    """두 모델을 모두 로드하고 해당 전처리 함수 및 크기를 반환합니다."""
    models = {}
    
    # 1. ConvNeXt 로드
    if os.path.exists(CONVNEXT_MODEL_PATH):
        try:
            models['ConvNeXt'] = {
                'model': tf.keras.models.load_model(CONVNEXT_MODEL_PATH), 
                'preprocess': convnext_preprocess, 
                'size': CONVNEXT_IMAGE_SIZE
            }
        except Exception as e:
            st.warning(f"⚠️ ConvNeXt 로드 실패 (호환성 오류): {e}", icon="⚠️")

    # 2. Xception 로드
    if os.path.exists(XCEPTION_MODEL_PATH):
        try:
            models['Xception'] = {
                'model': tf.keras.models.load_model(XCEPTION_MODEL_PATH), 
                'preprocess': xception_preprocess, 
                'size': XCEPTION_IMAGE_SIZE
            }
        except Exception as e:
            st.error(f"⚠️ Xception 모델 로드 실패. (오류: {e})", icon="🚨")
            
    if not models:
        st.error("🚨 치명적 오류: 유효한 모델 파일(.keras, .h5)을 찾을 수 없습니다.", icon="🚨")
        
    return models

def load_and_preprocess_image(image_file, target_size, preprocess_func):
    """업로드된 이미지를 로드하고, 모델에 맞는 전처리 함수를 사용하여 처리합니다."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return preprocess_func(img_array)

def get_safety_status(mushroom_name):
    """버섯 이름으로 독/식용 상태를 확인합니다."""
    return SAFETY_MAPPING.get(mushroom_name, "⚠️ 정보 부족 - 추가 확인 필요")

def predict_and_format_result(model_data, img_file):
    """단일 모델로 예측하고 결과를 TOP 3 형식으로 포맷합니다."""
    
    model = model_data['model']
    preprocess_func = model_data['preprocess']
    image_size = model_data['size']
    
    # 1. 전처리 
    processed_image = load_and_preprocess_image(img_file, image_size, preprocess_func)
    
    # 2. 예측 수행
    try:
        predictions = model.predict(processed_image, verbose=0) 
    except ValueError as e:
        st.warning(f"💡 예측 오류 발생. 모델 구조를 다시 확인하세요: {e}", icon="⚠️")
        return None, 0, "⚠️ 예측 실패", pd.DataFrame({'버섯 종류': ['오류'], '확신도 (%)': [0]})


    score = tf.nn.softmax(predictions[0]) 
    
    # 3. TOP 3 확신도 로직
    class_probabilities = list(zip(CLASS_NAMES, score.numpy()))
    sorted_probabilities = sorted(class_probabilities, key=lambda item: item[1], reverse=True)
    
    top_n = 3
    top_3_results = sorted_probabilities[:top_n]
    
    top_3_probs = [prob for name, prob in top_3_results]
    sum_top_3_probs = sum(top_3_probs)
    
    # 4. 결과 설정
    top_1_name = top_3_results[0][0]
    top_1_prob_original = top_3_results[0][1]
    
    if sum_top_3_probs > 0:
        # TOP 3 내에서 확신도 재정규화
        top_1_confidence = (top_1_prob_original / sum_top_3_probs) * 100
        safety_status = get_safety_status(top_1_name)
        chart_data = pd.DataFrame({
            '버섯 종류': [name for name, prob in top_3_results],
            '확신도 (%)': [(prob / sum_top_3_probs) * 100 for name, prob in top_3_results]
        })
    else:
        top_1_confidence = 0.00
        safety_status = "⚠️ 판독 불가능"
        chart_data = pd.DataFrame({'버섯 종류': ['판독 실패'], '확신도 (%)': [0]})
        
    return top_1_name, top_1_confidence, safety_status, chart_data

# --- Streamlit UI 시작 ---

st.set_page_config(page_title="통합 버섯 판독 시스템", layout="wide")

# 1. 배경 이미지 자동 로드 및 적용
BACKGROUND_IMAGE_BASE64 = get_base64_of_image(BACKGROUND_IMAGE_FILENAME)

if BACKGROUND_IMAGE_BASE64:
    set_custom_style(BACKGROUND_IMAGE_BASE64)
else:
    st.warning(f"⚠️ **{BACKGROUND_IMAGE_FILENAME}** 파일을 찾거나 로드할 수 없어 배경이 적용되지 않습니다. 파일을 확인해주세요.", icon="⚠️")


st.title("🍄 통합 ConvNeXt & Xception 판독 시스템")
st.caption("두 모델의 예측 결과를 비교하여 최종 안전 등급을 확인합니다.")

# 2. 모델 로드 (이곳에서 로드 메시지 출력)
all_models = load_all_models()

st.markdown("---") 

if all_models:
    # 3. 이미지 업로드
    uploaded_file = st.file_uploader("판독할 버섯 이미지를 업로드하세요 (.jpg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # 4. 레이아웃 분할
        col_img, col_info = st.columns([1, 2.5]) 
        
        with col_img:
            st.markdown("### 🖼️ 업로드 이미지")
            st.image(uploaded_file, use_container_width=True) 
            
            # 예측 버튼을 누르기 전까지는 예측 결과를 숨깁니다.
            if st.button("🍄 안전 판별 시작", use_container_width=True):
                st.session_state['run_prediction'] = True
        
        # 5. 예측 실행 및 결과 출력
        # 세션 상태를 사용하여 예측 버튼을 눌렀을 때만 실행되도록 합니다.
        if 'run_prediction' in st.session_state and st.session_state['run_prediction']:
            st.markdown("---")
            st.subheader("📊 예측 결과 비교")
            
            # 결과 컬럼 수 동적 생성
            result_columns = st.columns(len(all_models))
            
            for idx, (model_name, model_data) in enumerate(all_models.items()):
                
                with result_columns[idx]:
                    st.markdown(f"#### 🧠 {model_name} 모델 분석")
                    
                    # 예측 시작 시 스피너 표시
                    with st.spinner(f"{model_name} 예측 중 (입력: {model_data['size'][0]}x{model_data['size'][1]})..."):
                        
                        top_name, confidence, safety_status, chart_data = predict_and_format_result(model_data, uploaded_file)

                        if top_name: # 예측 성공 시에만 결과 출력
                            # 5-1. 최종 안전 판별 (가장 크게 강조)
                            if 'POISONOUS' in safety_status or '☠️' in safety_status:
                                st.error(f"🚨 최종 안전 판별: {safety_status}", icon="❌")
                            else:
                                st.success(f"✅ 최종 안전 판별: {safety_status}", icon="🍄")
                                
                            # 5-2. 메인 지표 출력
                            st.metric(label="예측된 버섯 종류", value=top_name)
                            st.metric(label="모델 확신도", value=f"{confidence:.2f}%")
                            
                            # 5-3. TOP 3 확신도 그래프 출력
                            st.markdown("##### 📈 TOP 3 확신도 분산")
                            st.bar_chart(chart_data, x='버섯 종류', y='확신도 (%)')
                        
                st.markdown("---")