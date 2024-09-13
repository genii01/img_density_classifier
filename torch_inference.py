import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np

# 모델 정의
class EfficientNetWithSize(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetWithSize, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.model._fc.in_features
        
        self.model._fc = nn.Sequential(
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

def load_model(model_path, num_classes):
    # 모델 초기화
    model = EfficientNetWithSize(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    # 이미지 전처리 및 원본 사이즈 추출
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 리사이즈
        transforms.ToTensor(),  # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
    ])
    
    image = Image.open(image_path).convert('RGB')  # RGB로 변환
    original_size = np.array(image.size, dtype=np.float32)  # 원본 이미지 사이즈 (width, height)
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image, original_size  # Numpy 배열과 원본 사이즈 반환

def infer(model, image, size):
    # GPU 사용 여부에 따라 모델과 데이터 전송
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    size = size.to(device)
    
    with torch.no_grad():
        outputs = model(image, size)
    
    return outputs

def main():
    model_path = 'best_model.pth'  # 저장된 모델 파일 경로
    image_path = '900_27913.png'  # 추론할 이미지 파일 경로
    num_classes = 5  # 모델의 클래스 수
    
    # 모델 로드
    model = load_model(model_path, num_classes)
    
    # 이미지 전처리 및 원본 사이즈 추출
    image, size = preprocess_image(image_path)
    size = torch.tensor([size], dtype=torch.float32)  # 배치 차원 추가
    
    # 추론 수행
    outputs = infer(model, image, size)
    
    # 예측 결과 출력
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    print(f"Predicted Class: {predicted_class}")
    print(f"Class Probabilities: {probabilities.cpu().numpy()}")

if __name__ == "__main__":
    main()
