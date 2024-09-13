import torch
import torch.onnx
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import numpy as np

class EfficientNetWithSize(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetWithSize, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        self.size_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(128 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, size):
        outputs = self.model(image)
        size_features = self.size_fc(size)
        combined_features = torch.cat((outputs, size_features), dim=1)
        logits = self.final_fc(combined_features)
        return logits

def load_model(model_path, num_classes=5):
    # 모델 초기화 및 가중치 로드
    model = EfficientNetWithSize(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    return model

def convert_model_to_onnx(model, dummy_input_image, dummy_input_size, onnx_path):
    # 모델을 ONNX로 변환
    torch.onnx.export(
        model,  # 변환할 모델
        (dummy_input_image, dummy_input_size),  # 더미 입력
        onnx_path,  # 저장할 ONNX 파일 경로
        input_names=['image', 'size'],  # 입력 텐서의 이름
        output_names=['output'],  # 출력 텐서의 이름
        dynamic_axes={
            'image': {0: 'batch_size'},  # 배치 크기 동적 설정
            'size': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    print(f"Model successfully converted to ONNX format and saved to {onnx_path}")

def main():
    model_path = 'best_model.pth'
    onnx_path = 'model.onnx'
    
    # 더미 입력 생성
    dummy_image = torch.randn(1, 3, 224, 224)  # 이미지 텐서
    dummy_size = torch.randn(1, 2)  # 사이즈 텐서
    
    # 모델 로드
    model = load_model(model_path, num_classes=5)
    
    # ONNX로 변환 및 저장
    convert_model_to_onnx(model, dummy_image, dummy_size, onnx_path)

if __name__ == "__main__":
    main()
