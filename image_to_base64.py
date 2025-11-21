import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# ⚠️ 이미지 파일 경로를 정확하게 입력하세요.
# 예시: forest_background.jpg
image_file_name = "forest-7406241_1280.jpg" 

base64_string = image_to_base64(image_file_name)
print(base64_string)

# ⬆️ 이 출력된 Base64 문자열을 아래 Streamlit 코드의 `BACKGROUND_IMAGE_BASE64` 변수에 붙여넣으세요.