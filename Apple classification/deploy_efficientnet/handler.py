import os
import json
import torch
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import torchvision.transforms as transforms
import io
from model import Efficientnet_AppleClassifier

class EfficientnetHandler(BaseHandler):
    def initialize(self, context):
        self.context = context
        self.manifest = context.manifest
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")
        self.model = self._load_model(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        image_size = (224, 224)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        images = []
        weight = None
        for item in data:
            if "data" in item:
                images.append(transform(Image.open(io.BytesIO(item["data"])).convert('RGB')).unsqueeze(0))
            if "weight" in item:
                weight = item["weight"]
        return torch.cat(images).to(self.device), torch.tensor([weight]).float().to(self.device)

    def inference(self, model_input):
        images, weight = model_input
        outputs = self.model(images, weight)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities

    def postprocess(self, inference_output):
        results = inference_output.cpu().detach().numpy().tolist()
        return [json.dumps({'predicted_class': str(result.index(max(result))), 'probabilities': result}) for result in results]

    def _load_model(self):
        model_path = ('../weight/Efficientnet_best_model.pth')
        cfg = {'dropout': 0.5}
        model = Efficientnet_AppleClassifier(cfg)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model