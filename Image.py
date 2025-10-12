import os
from PIL import Image

# 1. 이미지 파일이 있는 폴더와 저장할 폴더 경로를 설정하세요.
# 'your_mushroom_images' 대신 실제 이미지 폴더 이름을 입력하세요.
# 'resized_images'는 크기 조절된 이미지가 저장될 새로운 폴더입니다.
image_folder = 'my_mushroom_images'
output_folder = '붉은 뿔사슴버섯 X299'
target_size = (299, 299) # 목표 크기를 229x229로 설정

# 2. 크기 조절된 이미지를 저장할 폴더가 없으면 새로 만듭니다.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 설정한 이미지 폴더 안의 파일들을 하나씩 불러와서 처리합니다.
for filename in os.listdir(image_folder):
    # .jpg나 .png 확장자를 가진 파일만 처리합니다.
    if filename.endswith('.jpg') or filename.endswith('.png'):
        try:
            # 원본 이미지 파일의 전체 경로를 만듭니다.
            image_path = os.path.join(image_folder, filename)
            
            # 이미지를 엽니다.
            with Image.open(image_path) as img:
                # 이미지 크기를 224x224로 조절합니다.
                resized_img = img.resize(target_size)
                
                # 저장할 파일의 전체 경로를 만듭니다.
                output_path = os.path.join(output_folder, filename)
                
                # 크기 조절된 이미지를 새로운 폴더에 저장합니다.
                resized_img.save(output_path)
                
                print(f"{filename} 파일 크기 조절 및 저장 완료!")

        except Exception as e:
            # 오류가 발생한 경우 해당 파일명을 출력합니다.
            print(f"Error processing {filename}: {e}")

print("모든 이미지 처리 완료!")
