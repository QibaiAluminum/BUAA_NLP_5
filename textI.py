import torch
import torch.nn as nn
import re
import jieba
from gensim.models import Word2Vec
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(self, onehot_num):
        super(Net, self).__init__()
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)# 编码
        self.encode = torch.nn.Sequential(nn.Linear(onehot_size, embedding_size),nn.Dropout(0.5),nn.ReLU())# 解码
        self.decode = torch.nn.Sequential(nn.Linear(embedding_size, onehot_size),nn.Dropout(0.5),nn.Sigmoid())

    def forward(self, x):# 入
        em = self.encode(x).unsqueeze(dim=1)# 出
        out, (h, c) = self.lstm(em)
        res = 2*(self.decode(out[:,0,:])-0.5)
        return res

def dataR():
    with open('./data/test.txt', "r", encoding='utf-8') as f:
        file_read = f.readlines()
        textInput = ""
        for line in file_read:
            line = re.sub('\s', '', line)
            line = re.sub('，', '。', line)
            line = re.sub('！', '。', line)
            line = re.sub('？', '。', line)
            line = re.sub('[\u0000-\u3001]', '', line)
            line = re.sub('[\u3003-\u4DFF]', '', line)
            line = re.sub('[\u9FA6-\uFFFF]', '', line)
            textInput += line
    return textInput

text = dataR()
wordInText = []
for text_line in text.split('。'):
    seg_list = list(jieba.cut(text_line,cut_all=False))  # 使用精确模式
    if len(seg_list) < 5:
        seg_list.append("stopRead")
        wordInText.append(seg_list)
finalItera = torch.load(str(29) + ".pth")
model = Net(1024).eval().to(device)
model.load_state_dict(finalItera["models"])
vec_model = Word2Vec.load('model.model')

seqs = []
for sequence in wordInText:
    seqs += sequence

input_seq = torch.zeros(len(seqs), 1024).to(device)
result = ""
print("上文：")
print(text)

with torch.no_grad():
    for i in range(len(seqs)):
        input_seq[i] = torch.tensor(vec_model.wv[seqs[i]])
    end_num = 0
    length = 0
    while end_num < 10 and length < 100:
        out_res = model(input_seq.to(device))[-2:-1]
        key_value = vec_model.wv.most_similar(positive=np.array(out_res.cpu()), topn=30)
        print(key_value)
        key = key_value[np.random.randint(30)][0]
        if key == "stopRead":
            result += "。"
            end_num += 1
        else:
            result += key
        length += 1
        input_seq = torch.cat((input_seq, out_res), dim=0)
print("下文：")
print(result)
