# data_processor.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, IMAGE_FOLDER

def prepare_data_generators():
    """이미지 파일을 불러오고 학습 및 검증 데이터 생성기를 반환합니다."""

    # 1. ImageDataGenerator 설정
    # rescale=1./255: 픽셀 값을 0과 1 사이로 정규화 (CNN 모델의 표준)
    data_generator = ImageDataGenerator(
        rescale=1./255, 
        validation_split=VALIDATION_SPLIT, 
        # 선택 사항: 데이터 증강(Data Augmentation) 추가
        rotation_range=20, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        horizontal_flip=True
    )

    # 2. 훈련 데이터 생성기
    train_generator = data_generator.flow_from_directory(
        IMAGE_FOLDER,           # 이미지가 담긴 루트 폴더
        target_size=IMAGE_SIZE, # 299x299로 크기 재조정
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'       # 훈련 데이터만 사용
    )

    # 3. 검증 데이터 생성기
    validation_generator = data_generator.flow_from_directory(
        IMAGE_FOLDER,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'     # 검증 데이터만 사용
    )

    return train_generator, validation_generator