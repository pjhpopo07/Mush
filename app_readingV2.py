

# 2. 필요 라이브러리 임포트
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd


# 💡 [핵심 수정 1] Keras 백엔드를 TensorFlow로 명시 (Keras 3 호환성)
os.environ["KERAS_BACKEND"] = "tensorflow" 
import keras
print(f"Keras Backend: {os.environ.get('KERAS_BACKEND')}")

# --- Google Drive 마운트 ---
try:
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive 마운트 성공")
except Exception as e:
    print(f"Google Drive 마운트 실패: {e}")

# --- 모델별 전처리 함수 임포트 ---
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# ======================================================================
# ✅ [추가] Focal Loss 함수 정의
# ======================================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    focal_loss.__name__ = 'focal_loss'
    return focal_loss

# ======================================================================
# ❗ [사용자 설정] (Google Drive 경로)
# ======================================================================
XCEPTION_MODEL_PATH = '식용 이미지 추가한 CNN 모델.h5'
CONVNEXT_MODEL_PATH = 'convnext.keras'
XCEPTION_IMG_SIZE = (299, 299)
CONVNEXT_IMG_SIZE = (224, 224)

# ======================================================================
# ✅ 3. 데이터 폴더에서 클래스 이름 로드
# ======================================================================
IMAGE_FOLDER = 'your_mushroom_images'
try:
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {IMAGE_FOLDER}")
    all_items = os.listdir(IMAGE_FOLDER)
    class_names = sorted([d for d in all_items if os.path.isdir(os.path.join(IMAGE_FOLDER, d)) and not d.startswith('.')])
    if not class_names:
        raise FileNotFoundError(f"경로에 클래스 하위 폴더가 없습니다: {IMAGE_FOLDER}")
    print(f"✅ 데이터 폴더에서 클래스 로드 완료: {len(class_names)}개")
    print(f"클래스 목록: {class_names}")
except Exception as e:
    print(f"⚠️ 클래스 폴더({IMAGE_FOLDER}) 스캔 실패: {e}.")
    raise e

# ======================================================================
# ✅ 4. Custom Objects로 실제 모델 로드 (수정 적용)
# ======================================================================
try:
    print("--- 실제 모델 로드를 시작합니다 ---")
    
    # 💡 [핵심 수정 2] custom_objects에 Dense 및 Pooling 레이어를 명시적으로 추가
    custom_objects = {
        'focal_loss': categorical_focal_loss(gamma=2.0, alpha=0.25),
        'Dense': tf.keras.layers.Dense,
        'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D
    }
    
    # 1. Xception 모델 로드
    xception_model = tf.keras.models.load_model(
        XCEPTION_MODEL_PATH,
        custom_objects=custom_objects
    )
    
    # 2. ConvNeXt 모델 로드 (여기서 오류가 발생했었음)
    convnext_model = tf.keras.models.load_model(
        CONVNEXT_MODEL_PATH,
        custom_objects=custom_objects # 수정된 custom_objects 적용
    )
    
    print("✅ Xception, ConvNeXt 실제 모델 로드 성공!")
except Exception as e:
    print(f"⛔ [오류] 모델 로드 실패: {e}")
    # 모델 로드 실패 시, Gradio 앱 실행을 막기 위해 raise
    raise e

# ======================================================================
# ✅ 5. 예측 함수
# ======================================================================

def preprocess_image(image_pil, target_size, preprocess_function):
    image_resized = image_pil.resize(target_size, Image.LANCZOS)
    image_np = np.array(image_resized)
    # 이미지 채널 보정 (흑백/4채널 이미지 처리)
    if image_np.ndim == 2: image_np = np.stack([image_np]*3, axis=-1)
    elif image_np.shape[2] == 4: image_np = image_np[..., :3]
    image_batch = np.expand_dims(image_np, axis=0)
    return preprocess_function(image_batch.astype('float32'))

def format_predictions_for_label(predictions, class_names):
    """ Label(Dictionary) 형식으로 반환 """
    # Softmax 적용하여 확률로 변환 (모델의 최종 활성화 함수가 linear일 경우 필요)
    predictions = tf.nn.softmax(predictions).numpy()
    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return confidences

def predict_xception_model(image_pil):
    """ Xception 모델 실제 예측 """
    processed_batch = preprocess_image(image_pil, XCEPTION_IMG_SIZE, xception_preprocess)
    predictions = xception_model.predict(processed_batch, verbose=0)[0]
    return format_predictions_for_label(predictions, class_names)

def predict_convnext_model(image_pil):
    """ ConvNeXt 모델 실제 예측 """
    processed_batch = preprocess_image(image_pil, CONVNEXT_IMG_SIZE, convnext_preprocess)
    predictions = convnext_model.predict(processed_batch, verbose=0)[0]
    return format_predictions_for_label(predictions, class_names)

# --- 6. Gradio 인터페이스 메인 함수 ---
def compare_models(input_image):
    if input_image is None:
        return None, None, None, None

    image_pil = input_image

    xception_result = predict_xception_model(image_pil)
    convnext_result = predict_convnext_model(image_pil)

    # [이미지, Label결과, 이미지, Label결과] 반환
    return image_pil, xception_result, image_pil, convnext_result

# ======================================================================
# ✅ 7. Gradio UI 빌드 (Soft 테마)
# ======================================================================

# Apple의 파란색 링크/버튼과 깔끔한 느낌을 위해 Soft 테마 적용
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue")) as demo:

    gr.Markdown(
        """
        # 🍄 AI 버섯 분류기 비교
        **Xception** 모델과 **ConvNeXt** 모델의 성능을 비교합니다.
        """
    )

    # --- 1. 입력 영역 ---
    with gr.Group():
        gr.Markdown("### 1. 이미지 업로드")
        image_input = gr.Image(
            label="분석할 버섯 이미지를 업로드하세요.",
            type="pil",
            sources=['upload', 'clipboard', 'webcam'],
            height=400
        )

    gr.Markdown("---")

    gr.Markdown("### 2. 모델 비교 결과")

    # --- 2. 비교 결과 영역 ---
    with gr.Row(equal_height=True):

        # 2-1. 왼쪽 카드 (Xception)
        with gr.Group():
            gr.Markdown("## 1. Xception 모델")
            xception_image_output = gr.Image(label="분석 이미지", height=300)

            xception_result_output = gr.Label(
                label="분류 결과 (Top 3)",
                num_top_classes=3
            )

        # 2-2. 오른쪽 카드 (ConvNeXt)
        with gr.Group():
            gr.Markdown("## ✨ 2. ConvNeXt 모델")
            convnext_image_output = gr.Image(label="분석 이미지", height=300)

            convnext_result_output = gr.Label(
                label="분류 결과 (Top 3)",
                num_top_classes=3
            )

    # 4. 이벤트 연결
    image_input.change(
        fn=compare_models,
        inputs=[image_input],
        outputs=[
            xception_image_output, xception_result_output, 
            convnext_image_output, convnext_result_output
        ]
    )

# 8. 앱 실행
print("\n--- Gradio UI (Colab)을 실행합니다 ---")
demo.launch(debug=True)