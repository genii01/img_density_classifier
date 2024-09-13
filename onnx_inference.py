import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_onnx_model(model_path):
    # ONNX 모델 로드
    session = ort.InferenceSession(model_path)
    return session

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
    return image.numpy(), original_size  # Numpy 배열과 원본 사이즈 반환

def infer_onnx_model(session, image, size):
    # ONNX 모델로 추론 수행
    inputs = {
        'image': image,
        'size': size
    }
    outputs = session.run(None, inputs)
    return outputs[0]

def main():
    model_path = 'best_model.onnx'  # 변환된 ONNX 모델 파일 경로
    image_path = '900_27913.png'  # 추론할 이미지 파일 경로
    
    # 모델 로드
    session = load_onnx_model(model_path)
    
    # 이미지 전처리 및 원본 사이즈 추출
    image, size = preprocess_image(image_path)
    size = np.array([size], dtype=np.float32)  # 배치 차원 추가
    
    # 추론 수행
    output = infer_onnx_model(session, image, size)
    
    # 결과 출력
    print(f"Model output: {output}")

if __name__ == "__main__":
    main()
