# data_processor.py

import albumentations as A
from config import IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, IMAGE_FOLDER
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_augmentation_pipeline(target_size, is_train):
    """훈련 여부에 따라 Albumentations 파이프라인을 정의합니다."""
    
    # 공통: 이미지 크기 조정 및 정규화
    transforms = [
        A.Resize(target_size[0], target_size[1]),
        # Normalize: 0-255 값을 0-1로 정규화
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0), 
    ]
    
    if is_train:
        # 요청하신 6가지 증강 기법 (p=확률)
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # 버섯은 상하 대칭이 아닐 수 있으므로 확률을 낮춤
            A.GaussNoise(p=0.3),
            A.CoarseDropout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5), 
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
        ])

    # A.Compose는 이 모든 변환을 파이프라인으로 묶기
    return A.Compose(transforms)

class AlbumentationsDataSequence(tf.keras.utils.Sequence):
    """Albumentations를 사용하여 이미지를 로드하고 증강하는 커스텀 데이터 생성기"""
    
    def __init__(self, x_set, y_set, batch_size, augmentation_pipeline):
        self.x, self.y = x_set, y_set # x: 이미지 경로, y: 레이블
        self.batch_size = batch_size
        self.pipeline = augmentation_pipeline
        self.indices = np.arange(len(self.x))
        
        # 훈련 데이터에만 셔플 적용 (첫 번째 변환이 Resize/Normalize만 있다면 훈련이 아님)
        if len(augmentation_pipeline.transforms) > 2: 
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = self.y[batch_indices] 
        
        # 이미지 로드 및 증강 적용
        for i in batch_indices:
            img_path = self.x[i]
            # PIL로 이미지 로드 (RGB, 배열 형태)
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Albumentations 적용 (증강된 이미지가 메모리에서 실시간 생성됨)
            augmented = self.pipeline(image=image)
            batch_x.append(augmented['image'])
            
        return np.array(batch_x), np.array(batch_y)
    
    # 에포크가 끝날 때마다 데이터 인덱스를 다시 섞어줍니다.
    def on_epoch_end(self):
        if len(self.pipeline.transforms) > 2: 
            np.random.shuffle(self.indices)


# ----------------------------------------------------------------------
# 3. 데이터 로드 및 분할 (main.py에서 호출됨)
# ----------------------------------------------------------------------
def load_all_data(IMAGE_FOLDER):
    """모든 이미지 파일 경로와 레이블을 로드합니다."""
    # ... (생략, 기존 데이터 로드 및 레이블링 로직) ...
    # 이 함수는 이미지 경로와 원-핫 인코딩된 레이블을 반환합니다. 
    # (코드가 길어 생략하며, 기존 코드를 사용하거나 구현해야 합니다.)
    
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(IMAGE_FOLDER)) 
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(IMAGE_FOLDER, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_idx)
    
    num_classes = len(class_names)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    return np.array(image_paths), labels, num_classes


def prepare_data_generators():
    """모든 데이터를 로드하고 학습/검증 Sequence를 반환합니다."""
    
    # 1. 모든 이미지 경로와 레이블 로드
    all_image_paths, all_labels, num_classes = load_all_data(IMAGE_FOLDER)
    
    # 2. 학습/검증 데이터 분할
    x_train, x_val, y_train, y_val = train_test_split(
        all_image_paths, all_labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=all_labels
    )
    
    print(f"--- 데이터 로드 완료: 총 {len(all_image_paths)}개 이미지, {num_classes}개 클래스 ---")
    
    # 3. Albumentations 파이프라인 정의
    train_pipeline = get_augmentation_pipeline(IMAGE_SIZE, is_train=True)
    val_pipeline = get_augmentation_pipeline(IMAGE_SIZE, is_train=False) # 검증 데이터는 증강하지 않음 (Resize/Normalize만 적용)

    # 4. 데이터 Sequence 생성
    train_sequence = AlbumentationsDataSequence(
        x_train, y_train, BATCH_SIZE, train_pipeline
    )
    validation_sequence = AlbumentationsDataSequence(
        x_val, y_val, BATCH_SIZE, val_pipeline
    )
    
    return train_sequence, validation_sequence, num_classes
