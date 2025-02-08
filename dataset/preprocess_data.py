import os
import pandas as pd
import chardet
import re
# 假设txt文件存储在当前目录下的 'txt_files' 文件夹中
txt_folder = 'raw_datas'
csv_file = 'output.csv'

# 用来存储所有数据的列表
data = []

# 遍历所有的txt文件
for filename in os.listdir(txt_folder):
    if filename.endswith('.txt'):
        # 自动检测文件编码
        with open(os.path.join(txt_folder, filename), 'rb') as file:
            raw_data = file.read(10000)  # 读取前10000个字节
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # 使用检测到的编码打开文件
        try:
            with open(os.path.join(txt_folder, filename), 'r', encoding=encoding) as file:
                for line in file:
                    # 假设每行格式为 "sentence @sentiment"
                    parts = line.strip().split('@')
                    if len(parts) == 2:
                        sentence, sentiment = parts
                        #data.append([sentence, sentiment])
                        data.append([sentiment, sentence])
        except UnicodeDecodeError:
            # 如果出现解码错误，尝试使用忽略错误的方式
            with open(os.path.join(txt_folder, filename), 'r', encoding=encoding, errors='ignore') as file:
                for line in file:
                    parts = line.strip().split('@')
                    if len(parts) == 2:
                        sentence, sentiment = parts
                        #data.append([sentence, sentiment])
                        data.append([sentiment, sentence])

df = pd.DataFrame(data, columns=['Sentiment', 'News Headline'])
# 保存到CSV文件
# df.to_csv(csv_file, index=False, encoding='utf-8')

# print(f"Data has been saved to {csv_file}")

# 定义清理文本的函数
def clean_text(text):
    # 清理非ASCII字符
    if re.search(r'[^\x00-\x7F]+', text):
        return False  # 如果包含非法字符，返回False
    else:
        return True  # 否则返回True

# 读取CSV文件
# df = pd.read_csv('output.csv')

# 删除包含非法字符的行
df_cleaned = df[df['News Headline'].apply(clean_text)]  # 只保留合法的行

# 保存清理后的数据到新的CSV文件
df_cleaned.to_csv('all-data.csv', index=False, encoding='utf-8')

print(f"Cleaned data has been saved to 'all-data.csv'.")

