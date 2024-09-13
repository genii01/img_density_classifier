import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일에서 데이터 로드
csv_file = 'over_sampled_image_data.csv'
df = pd.read_csv(csv_file)

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')  # Convert to RGB
        label = row['label']
        width = row['image_width']
        height = row['image_height']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([width, height], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 모델 정의 (EfficientNet B4)
class EfficientNetWithSize(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetWithSize, self).__init__()
        self.model = efficientnet_b4(pretrained=True)
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

def generate_classification_report_and_roc_curve(true_labels, pred_probs, num_classes, output_dir='output'):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Classification Report
    class_report = classification_report(true_labels, np.argmax(pred_probs, axis=1), target_names=[f'class_{i}' for i in range(num_classes)])
    print("\nClassification Report:")
    print(class_report)
    
    # Save Classification Report to a text file
    with open(os.path.join(output_dir, 'classification_report_efficientnetb4_over_sample.txt'), 'w') as f:
        f.write(class_report)
    
    # ROC Curve
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(f'{output_dir}/roc_curve_efficientnetb4_over_sample.png')
    plt.show()

def train_and_evaluate(num_epochs=10, batch_size=32, learning_rate=1e-4, num_classes=5, model_save_path='best_model.pth', output_dir='output', patience=5):
    # 데이터 로드 및 전처리 설정
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),    # 랜덤 크롭
        transforms.RandomHorizontalFlip(),    # 랜덤 수평 플립
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상 변형
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
    ])

    # 전체 데이터셋 준비
    dataset = CustomDataset(df, transform=transform)

    # 학습 및 검증 데이터셋 분리
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 학습 및 검증 데이터로더 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU (MPS)
        print("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')  # CUDA GPU
        print("Using CUDA (GPU)")
    else:
        device = torch.device('cpu')   # Fallback to CPU
        print("Using CPU")

    model = EfficientNetWithSize(num_classes=num_classes).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ReduceLROnPlateau 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 최적 모델을 저장하기 위한 변수 및 Early Stopping 설정
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
            for images, sizes, labels in train_loader:
                images = images.to(device)
                sizes = sizes.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images, sizes)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'Loss': running_loss / (pbar.n + 1)})
                pbar.update(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.4f}")

        # 검증 루프
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validation", unit='batch') as pbar:
                for images, sizes, labels in val_loader:
                    images = images.to(device)
                    sizes = sizes.to(device)
                    labels = labels.to(device)

                    outputs = model(images, sizes)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    pbar.update(1)

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(outputs.cpu().detach().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

        # ReduceLROnPlateau 스케줄러 호출
        scheduler.step(avg_val_loss)

        # 가장 좋은 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset early stopping counter
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with Validation Loss: {avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    # Classification Report 및 ROC Curve 작성
    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)
    generate_classification_report_and_roc_curve(all_val_labels, all_val_preds, num_classes, output_dir=output_dir)

# 하이퍼파라미터 설정 및 학습 실행
num_epochs = 30
batch_size = 128
learning_rate = 1e-4
num_classes = 5 
model_save_path = 'best_efficientnetb4_model_over_sample.pth'
output_dir = 'output'
patience = 5  # Early stopping patience

train_and_evaluate(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, num_classes=num_classes, model_save_path=model_save_path, output_dir=output_dir, patience=patience)
