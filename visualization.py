import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일에서 데이터 로드
csv_file = 'image_data.csv'
df = pd.read_csv(csv_file)

# matplotlib 및 seaborn 설정
plt.figure(figsize=(14, 6))

# 서브플롯 생성
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='label', y='image_height')
plt.title('Image Height Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Image Height')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='label', y='image_width')
plt.title('Image Width Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Image Width')

# 그림 저장
plt.tight_layout()
plt.savefig('image_size_distribution_by_label.png')
plt.show()
