import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torchvision import transforms

# 모델과 feature extractor 로드
model = AutoModelForImageClassification.from_pretrained("./saved_model")
model.eval()  # 모델을 평가 모드로 전환 (dropout, batchnorm 비활성화)

feature_extractor = AutoFeatureExtractor.from_pretrained("google/efficientnet-b0")

# 테스트 이미지 로드 및 전처리
image_path = "test_image.jpg"  # 추론할 이미지 경로
image = Image.open(image_path)

# 이미지 크기 정보 얻기
height, width = image.size

# 이미지 전처리 (모델과 학습 과정에서 사용한 전처리와 동일)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 전처리된 이미지
input_image = transform(image).unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)

# height/width를 텐서로 변환 (배치 차원 추가)
height_tensor = torch.tensor([height], dtype=torch.float32)
width_tensor = torch.tensor([width], dtype=torch.float32)

# 추론 수행
with torch.no_grad():  # 추론 시에는 gradients 계산 비활성화
    outputs = model(pixel_values=input_image, height=height_tensor, width=width_tensor)
    logits = outputs.logits

# 클래스 예측
predicted_class = torch.argmax(logits, dim=-1).item()

# 클래스 결과 출력 (예: CIFAR-10의 5가지 클래스 중 예측된 클래스)
class_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]
print(f"Predicted class: {class_names[predicted_class]}")