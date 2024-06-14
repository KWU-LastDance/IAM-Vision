import os
import torch
import torch.nn as nn
import torchvision.models as models
import os

class Densenet_AppleClassifier(nn.Module):
    def __init__(self):
        super(Densenet_AppleClassifier, self).__init__()
        base_model = models.densenet121(pretrained=True)
        # DenseNet121의 마지막 분류층을 제외한 나머지 부분을 사용
        self.feature_extractor = nn.Sequential(*list(base_model.features), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # DenseNet121의 출력 특징 차원 확인
        self.num_features = base_model.classifier.in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=0.5)

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=0.5)

        # 특징과 무게 정보를 결합한 뒤 클래스를 분류
        self.classifier = nn.Linear(self.num_features * 6 + 10, 3)  # 6개 이미지의 특징과 무게 특징 결합

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
    model = Densenet_AppleClassifier()
    model_path =('../weight/densenet_best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model