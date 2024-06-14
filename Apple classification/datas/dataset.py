import os
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from typing import Type, Tuple

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_dir, transform=None):
        self.image_data = []
        self.transform = transform
        # 레이블 매핑
        self.label_map = {'특': 0, '상': 1, '보통': 2}

        self.collect_data(root_dir, csv_dir)

    def collect_data(self, root_dir, csv_dir):
    # 모든 폴더 및 하위 폴더 탐색
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for subfolder_name in os.listdir(folder_path):
                    images = []
                    weights = []
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    csv_path = os.path.join(csv_dir, f"{folder_name}_label.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        # 파일 이름을 맨뒤 숫자로 정렬
                        img_files = sorted(os.listdir(subfolder_path), key=lambda x: int(x.split('-')[-1].split('.')[0]))
                        for img_file in img_files:
                            # 이미지 이름에 맞는 png가 있을 경우
                            if img_file.endswith('.png'):
                                img_path = os.path.join(subfolder_path, img_file)
                                img_name_without_ext = os.path.splitext(img_file)[0]

                                # df에서 이름과 같을 경우
                                row = df[df['Name'] == img_name_without_ext]
                                if not row.empty:
                                    images.append(img_path)
                                    weight = row['Weight'].iloc[0]
                                    label = row['label'].iloc[0]
                                    numeric_label = self.label_map.get(label, -1)
                                    weights.append(weight)
                        if images:
                            self.image_data.append({
                                'images': images,
                                'label': numeric_label,
                                'weight': sum(weights) / len(weights)  # 평균 무게 사용
                            })


    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        images = [self.transform(Image.open(img).convert('RGB')) for img in item['images']]
        return torch.stack(images), item['label'], item['weight']


def make_dataset(image_dir: str, csv_dir: str, image_size: Tuple[int, int]) -> Type[torch.utils.data.Dataset]:
    """
    Make pytorch Dataset for given task.
    Read the image using the PIL library and return it as an np.array.

    Args:
        image_dir (str): dataset directory
        csv_dir (str): dataset directory
        image_size (Tuple[int, int]): Target size of the images (height, width)

    Returns:
        torch.Dataset: pytorch Dataset
    """

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(root_dir=image_dir,
                            csv_dir=csv_dir,
                            transform=transform)

    return dataset
