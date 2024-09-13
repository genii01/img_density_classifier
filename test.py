import pandas as pd 


df = pd.read_csv('./image_data.csv')
res = df['label'].value_counts()
print(res)

# df_tmp=  df[df['label'] == 4]

# df_tmp = df_tmp[df_tmp['image_height'] >= 1000]
# print(df_tmp['image_height'].tolist())
