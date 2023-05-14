import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.embeddings[idx]


def load_embeddings(file_path):
    with open(file_path, "r") as f:
        # 读取第一行，获取向量数量和维度
        n, m = map(int, f.readline().strip().split())
        # 创建空数组
        embeddings = np.zeros((n, m))
        index_to_id = {}
        # 读取每一行向量数据
        for i, line in enumerate(f):
            parts = line.strip().split()
            # 第一列是id，之后的列是向量值
            index_to_id[i] = parts[0]
            embeddings[i] = [float(p) for p in parts[1:]]
        # 转换为 PyTorch 张量
        embeddings = torch.from_numpy(embeddings).float()
        return embeddings, index_to_id


def save_embeddings(embeddings, root_path, original_file_name, index_to_id):
    # 从文件名中解析dim，length，num信息
    file_name = os.path.basename(original_file_name)
    file_info = file_name.split(".")[0].split("_")
    dim = int(file_info[1][1:])
    length = int(file_info[2][1:])
    num = int(file_info[3][1:])
    reduced_file_name = f"dw_d{dim}_l{length}_n{num}_reduced.emb"
    reduced_file_path = os.path.join(root_path, reduced_file_name)
    with open(reduced_file_path, "w") as f:
        # 写入向量数量和维度
        n, m = embeddings.shape
        f.write(f"{n} {m}\n")
        # 写入每一行向量数据
        for i in range(n):
            embedding = embeddings[i]
            id = index_to_id[i]
            # 第一列是id，之后的列是向量值
            line = f"{id} " + " ".join(str(x) for x in embedding.tolist()) + "\n"
            f.write(line)


# 加载数据集
EMB_ROOT_PATH = '../data/emb/'
data_dir = os.path.join(EMB_ROOT_PATH, 'deepwalk')
output_dir = os.path.join(EMB_ROOT_PATH, 'encoded')
files = os.listdir(data_dir)
embeddings = {}
file_index_to_id = {}
for file in files:
    if not file.endswith(".emb"):
        continue
    file_path = os.path.join(data_dir, file)
    embeddings[file], file_index_to_id[file] = load_embeddings(file_path)

# 对每个嵌入向量文件进行降维
for file, x in embeddings.items():

    input_dim = x.shape[1]
    if input_dim != 64:
        output_dim = 64

        model = Autoencoder(input_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 创建数据集和数据加载器
        dataset = EmbeddingsDataset(x)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 训练模型
        for epoch in range(100):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # 保存模型
        encoded_x = model.encoder(torch.tensor(x, dtype=torch.float))
        save_embeddings(encoded_x, output_dir, file, file_index_to_id[file])
    else:
        save_embeddings(x, output_dir, file, file_index_to_id[file])
    # torch.save(model.state_dict(), f"autoencoder_dw_d{dim}_l{length}_n{num}.pt")
    #
    # # 保存降维后的向量
    # with torch.no_grad():
    #     encoded_x = model.encoder(torch.tensor(x, dtype=torch.float))
    #     np.savetxt(f"autoencoder_dw_d{dim}_l{length}_n{num}_encoded.txt", encoded_x.numpy())
