import os
import pandas as pd
from PIL import Image

# 이미지 파일이 저장된 디렉토리
base_dir = './img_data'

# 폴더에 해당하는 label을 정의
folder_to_label = {
    '100': 0,
    '300': 1,
    '500': 2,
    '700': 3,
    '900': 4
}

# 결과를 저장할 리스트
data = []

# 모든 폴더를 순회
for folder, label in folder_to_label.items():
    folder_path = os.path.join(base_dir, folder)
    
    # 폴더 내의 모든 파일을 순회
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            
            # 이미지 열기
            with Image.open(file_path) as img:
                image_width, image_height = img.size
            
            # 데이터 추가
            data.append({
                'index': len(data),
                'file_path': file_path,
                'filename': filename,
                'image_height': image_height,
                'image_width': image_width,
                'label': label
            })

# DataFrame으로 변환
df = pd.DataFrame(data)

# CSV 파일로 저장
df.to_csv('image_data.csv', index=False)

print("CSV 파일이 성공적으로 생성되었습니다.")
