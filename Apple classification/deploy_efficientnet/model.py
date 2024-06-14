import os
import torch
import torch.nn as nn
import torchvision.models as models
import os

class Efficientnet_AppleClassifier(nn.Module):
    def __init__(self):
        super(Efficientnet_AppleClassifier, self).__init__()
        # EfficientNet-B0 모델을 pretrained 가중치로 로드
        base_model = models.efficientnet_b0(pretrained=True)
        
        # EfficientNet의 마지막 분류층을 제외한 나머지 레이어들을 사용
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-2], 
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten()
        )
        
        # EfficientNet-B0의 출력 특성 차원
        self.num_features = base_model.classifier[1].in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=0.5)

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=0.5)

        # 특징과 무게 정보를 결합한 뒤 클래스를 분류
        self.classifier = nn.Linear(self.num_features * 6 + 10, 3)  # 6개 이미지의 특징 결합

    def forward(self, x, weights):
        batch_size = x.size(0)
        features = [self.feature_extractor(x[:, i, :, :, :]) for i in range(6)]
        combined_features = torch.cat(features, dim=1)
        combined_features = self.dropout(combined_features)
        weight_features = self.weight_processor(weights.unsqueeze(2))
        weight_features = self.weight_dropout(weight_features)
        weight_features = weight_features.view(batch_size, -1)
        combined_features = torch.cat((combined_features, weight_features), dim=1)
        output = self.classifier(combined_features)
        return output
    
    
def model_fn():
    model = Efficientnet_AppleClassifier()
    model_path =('../weight/efficientnet_best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model