import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
sys.path.append(os.path.abspath('..'))


class Efficientnet_AppleClassifier(nn.Module):
    def __init__(self,cfg):
        super(Efficientnet_AppleClassifier, self).__init__()
        # torchvision 0.8.0 이상에서 사용 가능
        base_model = models.efficientnet_b0(pretrained=True)
        
        # EfficientNet의 마지막 분류층을 제외한 나머지를 사용
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2], nn.AdaptiveAvgPool2d(1), nn.Flatten())
        
        # EfficientNet-B0의 출력 특성 차원
        self.num_features = base_model.classifier[1].in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=cfg['dropout'])

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=cfg['dropout'])

        # 특징과 무게 정보를 결합한 뒤 클래스를 분류
        self.classifier = nn.Linear(self.num_features * 6 + 10, 3)  # 6개 이미지의 특징 결합

    def forward(self, x, weights):
        features = [self.feature_extractor(img) for img in x]
        combined_features = torch.cat(features, dim=1)
        
        # 이미지 특징에 드롭아웃 적용
        combined_features = self.dropout(combined_features)
        
        weight_features = self.weight_processor(weights.unsqueeze(1))
        
        # 무게 특징에 드롭아웃 적용
        weight_features = self.weight_dropout(weight_features)

        # 특징 결합
        combined_features = torch.cat((combined_features, weight_features), dim=1)
        
        output = self.classifier(combined_features)
        return output


class Densenet_AppleClassifier(nn.Module):
    def __init__(self,cfg):
        super(Densenet_AppleClassifier, self).__init__()
        base_model = models.densenet121(pretrained=True)
        # DenseNet121의 마지막 분류층을 제외한 나머지 부분을 사용
        self.feature_extractor = nn.Sequential(*list(base_model.features), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # DenseNet121의 출력 특징 차원 확인
        self.num_features = base_model.classifier.in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=cfg['dropout'])

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=cfg['dropout'])

        # 특징과 무게 정보를 결합한 뒤 클래스를 분류
        self.classifier = nn.Linear(self.num_features * 6 + 10, 3)  # 6개 이미지의 특징과 무게 특징 결합

    def forward(self, x, weights):
        features = [self.feature_extractor(img) for img in x]
        combined_features = torch.cat(features, dim=1)

        # 이미지 특징에 드롭아웃 적용
        combined_features = self.dropout(combined_features)

        weight_features = self.weight_processor(weights.unsqueeze(1))

        # 무게 특징에 드롭아웃 적용
        weight_features = self.weight_dropout(weight_features)

        # 특징 결합
        combined_features = torch.cat((combined_features, weight_features), dim=1)

        output = self.classifier(combined_features)
        return output
    
    
class Resnet_AppleClassifier(nn.Module):
    def __init__(self,cfg):
        super(Resnet_AppleClassifier, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())
        
        # feature_extractor의 출력 차원을 확인
        self.num_features = base_model.fc.in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=cfg['dropout'])  # 드롭아웃 확률을 50%로 설정

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=cfg['dropout'])  # 무게 특징에 대한 드롭아웃

        # 특징과 무게 정보를 결합한 뒤 클래스를 분류
        self.classifier = nn.Linear(self.num_features * 6 + 10, 3)  # 6개 이미지의 특징 결합

    def forward(self, x, weights):
        features = [self.feature_extractor(img) for img in x]
        combined_features = torch.cat(features, dim=1)
        
        # 이미지 특징에 드롭아웃 적용
        combined_features = self.dropout(combined_features)
        
        weight_features = self.weight_processor(weights.unsqueeze(1))
        
        # 무게 특징에 드롭아웃 적용
        weight_features = self.weight_dropout(weight_features)

        # 특징 결합
        combined_features = torch.cat((combined_features, weight_features), dim=1)
        
        output = self.classifier(combined_features)
        return output
    

