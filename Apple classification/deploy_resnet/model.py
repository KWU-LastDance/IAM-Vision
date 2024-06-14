import os
import torch
import torch.nn as nn
import torchvision.models as models
import os

class Resnet_AppleClassifier(nn.Module):
    def __init__(self,cfg):
        super(Resnet_AppleClassifier, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())
        
        # feature_extractor의 출력 차원을 확인
        self.num_features = base_model.fc.in_features

        # 드롭아웃 레이어 추가
        self.dropout = nn.Dropout(p=0.5)  # 드롭아웃 확률을 50%로 설정

        # 무게 데이터 처리를 위한 선형 레이어 추가
        self.weight_processor = nn.Linear(1, 10)
        self.weight_dropout = nn.Dropout(p=0.5)  # 무게 특징에 대한 드롭아웃

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
    
    
def model_fn():
    model = Resnet_AppleClassifier()
    model_path =('../weight/resnet_best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model