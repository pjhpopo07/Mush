import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
import numpy as np
import os
from PIL import Image
import pandas as pd
from collections import OrderedDict # 결과 순서 유지를 위해 사용

# ======================================================================
# 1. [사용자 설정] (VS Code/로컬 경로 기준)
# ======================================================================
# ⚠️ 파일 경로를 로컬에 맞게 수정하세요.
XCEPTION_MODEL_PATH = '식용 이미지 추가한 CNN.h5' # H5 또는 .keras 파일 경로
CONVNEXT_MODEL_PATH = 'convNext.keras'
XCEPTION_IMG_SIZE = (299, 299)
CONVNEXT_IMG_SIZE = (224, 224)

# ⚠️ 데이터 폴더에서 로드한 클래스 이름 (순서 절대 일치)
CLASS_NAMES = ['개암버섯', '노란개암버섯', '독우산광대버섯', '마귀광대버섯', '맑은애주름버섯', 
               '붉은 뿔사슴버섯', '붉은점박이광대버섯', '영지버섯', '졸각버섯', '흰주름버섯'] 

<<<<<<< HEAD
# ======================================================================
# 2. 모델 로드 및 유틸리티 함수
# ======================================================================
=======
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
>>>>>>> 07e5b37738f25cf6523f37a8e4a24d20b99b92b0

@st.cache_resource
def load_all_models_and_prep():
    """두 모델을 모두 로드하고 해당 전처리 함수를 반환합니다."""
    models = OrderedDict()
    custom_loss = {'focal_loss': lambda y_true, y_pred: tf.reduce_mean(-y_true * tf.math.log(y_pred))} # Focal Loss 호환성 함수

    # ConvNeXt 로드
    if os.path.exists(CONVNEXT_MODEL_PATH):
        try:
            models['ConvNeXt'] = {
                'model': tf.keras.models.load_model(CONVNEXT_MODEL_PATH, custom_objects=custom_loss),
                'preprocess': convnext_preprocess, 
                'size': CONVNEXT_IMG_SIZE
            }
            st.success("✅ ConvNeXt 모델 로드 완료.", icon="✅")
        except Exception as e:
            st.warning(f"⚠️ ConvNeXt 로드 실패 (호환성 오류): {e}", icon="⚠️")

    # Xception 로드
    if os.path.exists(XCEPTION_MODEL_PATH):
        try:
            models['Xception'] = {
                'model': tf.keras.models.load_model(XCEPTION_MODEL_PATH, custom_objects=custom_loss),
                'preprocess': xception_preprocess, 
                'size': XCEPTION_IMG_SIZE
            }
            st.success("✅ Xception 모델 로드 완료.", icon="✅")
        except Exception as e:
            st.error(f"⚠️ Xception 모델 로드 실패. (오류: {e})", icon="🚨")
            
    return models

def preprocess_image_for_model(image_pil, target_size, preprocess_function):
    """PIL 이미지를 받아 모델 입력 텐서로 전처리합니다."""
    # (이미지 채널/크기 조정 로직)
    image_resized = image_pil.resize(target_size, Image.LANCZOS)
    image_np = np.array(image_resized)
    if image_np.ndim == 2: image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[2] == 4: image_np = image_np[..., :3]
    image_batch = np.expand_dims(image_np, axis=0)
    
    # 모델별 전처리 함수 적용
    return preprocess_function(image_batch.astype('float32'))

def predict_and_format(model_data, image_pil):
    """단일 모델로 예측하고 Gradio Label 형식(Dict)으로 결과를 포맷합니다."""
    model = model_data['model']
    preprocess_func = model_data['preprocess']
    target_size = model_data['size']
    
    processed_batch = preprocess_image_for_model(image_pil, target_size, preprocess_func)
    
    # 예측 수행 및 Softmax 적용
    predictions = model.predict(processed_batch)[0]
    score = tf.nn.softmax(predictions).numpy()
    
    # Label 컴포넌트용 Dictionary 생성
    confidences = {CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))}
    return confidences


# ======================================================================
# 3. Streamlit UI (Gradio Layout 흉내)
# ======================================================================

st.set_page_config(page_title="CNN 모델 비교 분석", layout="wide")
st.title("🍄 AI 버섯 분류기 비교 (ConvNeXt vs. Xception)")
st.caption("업로드된 이미지를 두 모델이 분석하여 결과를 비교합니다.")
st.markdown("---")

# 1. 모델 로드
all_models = load_all_models_and_prep()

if all_models:
    # 2. 이미지 업로드 섹션
    gr_group_container = st.container(border=True)
    with gr_group_container:
        st.markdown("### 1. 이미지 업로드")
        uploaded_file = st.file_uploader(
            "분석할 버섯 이미지를 업로드하세요.", 
            type=["jpg", "jpeg", "png"]
        )

    # 3. 비교 결과 섹션
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 2. 모델 비교 결과")
        
        # 파일 업로드 객체를 PIL Image 객체로 변환
        image_pil = Image.open(uploaded_file).convert("RGB")

        # 4. 결과 출력 (Gradio의 Row/Group Layout 모방)
        col_xception, col_convnext = st.columns(2, gap="large")
        
        # --- 4-1. Xception 결과 ---
        with col_xception:
            st.markdown("## 1. Xception 모델")
            
            # 1. 분석 이미지 출력
            st.image(image_pil, caption="분석 이미지", use_container_width=True)
            
            # 2. 예측 및 결과 출력
            with st.spinner("Xception 예측 중..."):
                xception_result_dict = predict_and_format(
                    all_models.get('Xception'), 
                    image_pil
                )
                if xception_result_dict:
                    st.markdown("#### 분류 결과 (Top 3)")
                    # Gradio Label 컴포넌트처럼 결과를 깔끔하게 테이블로 출력
                    df_xception = pd.DataFrame(xception_result_dict.items(), columns=['버섯 종류', '확신도']).sort_values(by='확신도', ascending=False).head(3)
                    df_xception['확신도'] = (df_xception['확신도'] * 100).apply(lambda x: f"{x:.2f}%")
                    st.dataframe(df_xception, hide_index=True, use_container_width=True)
                else:
                    st.error("Xception 모델이 로드되지 않았습니다.")


        # --- 4-2. ConvNeXt 결과 ---
        with col_convnext:
            st.markdown("## ✨ 2. ConvNeXt 모델")

            # 1. 분석 이미지 출력
            st.image(image_pil, caption="분석 이미지", use_container_width=True)

            # 2. 예측 및 결과 출력
            with st.spinner("ConvNeXt 예측 중..."):
                convnext_result_dict = predict_and_format(
                    all_models.get('ConvNeXt'), 
                    image_pil
                )
                if convnext_result_dict:
                    st.markdown("#### 분류 결과 (Top 3)")
                    df_convnext = pd.DataFrame(convnext_result_dict.items(), columns=['버섯 종류', '확신도']).sort_values(by='확신도', ascending=False).head(3)
                    df_convnext['확신도'] = (df_convnext['확신도'] * 100).apply(lambda x: f"{x:.2f}%")
                    st.dataframe(df_convnext, hide_index=True, use_container_width=True)
                else:
                    st.error("ConvNeXt 모델이 로드되지 않았습니다.")