# import pandas as pd

# # 데이터 파일 경로
# input_file_path = 'image_data.csv'
# output_file_path = 'sampled_image_data.csv'

# # CSV 파일 읽기
# df = pd.read_csv(input_file_path)

# # label별로 최대 1000건 샘플링
# def sample_label_group(df, label, n_samples=1000):
#     label_group = df[df['label'] == label]
#     if len(label_group) > n_samples:
#         label_group = label_group.sample(n=n_samples, random_state=1)
#     return label_group

# # 샘플링된 데이터프레임 생성
# sampled_df = pd.concat([sample_label_group(df, label) for label in df['label'].unique()])

# # 샘플링된 데이터프레임을 CSV 파일로 저장
# sampled_df.to_csv(output_file_path, index=False)

# print(f"샘플링된 데이터가 '{output_file_path}'에 저장되었습니다.")

import pandas as pd

# 데이터 파일 경로
input_file_path = 'image_data.csv'
output_file_path = 'over_sampled_image_data.csv'

# CSV 파일 읽기
df = pd.read_csv(input_file_path)

# label별로 최대 3000건 샘플링 및 부족 시 oversampling
def sample_label_group(df, label, n_samples=1000):
    label_group = df[df['label'] == label]
    if len(label_group) >= n_samples:
        label_group = label_group.sample(n=n_samples, random_state=1)
    else:
        # 부족한 샘플 개수를 채우기 위해 oversampling
        oversampled_label_group = label_group.sample(n=n_samples, replace=True, random_state=1)
        label_group = oversampled_label_group
    return label_group

# 샘플링된 데이터프레임 생성
sampled_df = pd.concat([sample_label_group(df, label) for label in df['label'].unique()])

# 샘플링된 데이터프레임을 CSV 파일로 저장
sampled_df.to_csv(output_file_path, index=False)

print(f"샘플링된 데이터가 '{output_file_path}'에 저장되었습니다.")