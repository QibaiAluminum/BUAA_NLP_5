import torch
import torch.nn as nn
import re
import jieba
from tqdm import trange
from gensim.models import Word2Vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(self, onehot_num):
        super(Net, self).__init__()
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)
        self.encode = torch.nn.Sequential(nn.Linear(onehot_size, embedding_size),nn.Dropout(0.5),nn.ReLU())
        self.decode = torch.nn.Sequential(nn.Linear(embedding_size, onehot_size),nn.Dropout(0.5),nn.Sigmoid())

    def forward(self, x):
        em = self.encode(x).unsqueeze(dim=1)
        out, (h, c) = self.lstm(em)
        res = 2*(self.decode(out[:,0,:])-0.5)
        return res

def dataR():
    with open('./data/射雕英雄传.txt', "r", encoding='utf-8') as f:
        file_read = f.readlines()
        dataO = ""
        for line in file_read:
            # 由于标点符号过多，生成训练效果不好，所以将所有结束的标点符号，比如感叹号问号全部替换成句号
            line = re.sub('！', '。', line)
            line = re.sub('，', '。', line)
            line = re.sub('？', '。', line)
            line = re.sub('\s','', line)
            line = re.sub('[\u0000-\u3001]', '', line)
            line = re.sub('[\u3003-\u4DFF]', '', line)
            line = re.sub('[\u9FA6-\uFFFF]', '', line)
            if len(line) < 200:
                continue
            dataO += line
        f.close()
    return dataO
dataO = dataR()
wordInText = list()
for textL in dataO.split('。'):
    segmentL = list(jieba.cut(textL, cut_all=False))  # 使用精确模式
    if len(segmentL) < 5:
        continue
    segmentL.append("stopRead")
    wordInText.append(segmentL)

# 获得word2vec模型
model = Word2Vec(sentences=wordInText,sg=0, vector_size=1024, min_count=1, window=10, epochs=10)
model.save('model.model')
# lstm训练
sequences = wordInText
vectorInM = Word2Vec.load('model.model')
model = Net(1024).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

for epochID in range(30):
    for i in trange(0, len(sequences) // 10 - 1):
        seq = []
        for j in range(10):
            seq += sequences[i + j]
        target = []
        for h in range(10):
            target += sequences[i + 10 + h]
        seqInput = torch.zeros(len(seq), 1024)
        for m in range(len(seq)):
            seqInput[m] = torch.tensor(vectorInM.wv[seq[m]])
            seqTar = torch.zeros(len(target), 1024)
        for n in range(len(target)):
            seqTar[n] = torch.tensor(vectorInM.wv[target[n]])
        seqTotal = torch.cat((seqInput, seqTar), dim=0)
        optimizer.zero_grad()
        resOutput = model(seqTotal[:-1].to(device))
        funcA = ((resOutput[-seqTar.shape[0]:] ** 2).sum(dim=1)) ** 0.5
        funcB = ((seqTar.to(device) ** 2).sum(dim=1)) ** 0.5
        lossFun = (1 - (resOutput[-seqTar.shape[0]:] * seqTar.to(device)).sum(dim=1) / funcA / funcB).mean()
        lossFun.backward()
        optimizer.step()
        if i % 50 == 0:
            print("ID: ", epochID,"LossFun: ", lossFun.item(), " Res: ",resOutput[-seqTar.shape[0]:].max(dim=1).indices, seqTar.max(dim=1).indices)
    state = {"models": model.state_dict()}
    torch.save(state, str(epochID) + ".pth")
