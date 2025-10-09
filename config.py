# config.py

# 이미지 및 모델 설정
IMAGE_SIZE = (299, 299)
NUM_CHANNELS = 3 # RGB 이미지이므로 3
NUM_CLASSES = 2  # 분류할 버섯 종류의 수에 맞게 변경하세요!

# 데이터 경로 및 준비
IMAGE_FOLDER = 'your_mushroom_images' # 원본 이미지가 있는 폴더
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# 훈련 설정
EPOCHS = 10 
LEARNING_RATE = 0.0001