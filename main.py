import os
import pandas as pd
import numpy as np
from src import ppi
from src import classifiers as cl
from sklearn.model_selection import train_test_split


def parse_emb(ge_method, file_name):
    file_path = os.path.join(EMB_ROOT_PATH, ge_method, file_name)
    data = file_name.strip('.emb').split('_')
    param = {i[0]: i[1:] for i in data[1:]}

    with open(file_path, 'r') as f:
        data = [line.strip().split() for line in f.readlines()[1:]]
    X = [line[1:] for line in data]
    X = np.asarray(X, dtype=float)
    target = [1 if line[0] in positive_gene_id_set else 0 for line in data]
    target = np.asarray(target, dtype=int)
    return param, X, target


def training(emb_path, cl_method, ge_method):
    filenames = os.listdir(emb_path)
    results = []
    result_file_path = 'result/' + ge_method + '_' + cl_method + '.xls'
    for name in filenames:
        param, X, y = parse_emb(ge_method=ge_method, file_name=name)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        acc, f1, auc = cl.nb(x_train, x_test, y_train, y_test)

        if ge_method == 'deepwalk':
            results.append([param['d'], param['l'], param['n'], acc, f1, auc])
            print('dim={},length={},num={},accuracy={}，f1-score={},auc={}'
                  .format(param['d'], param['l'], param['n'], acc, f1, auc))
        elif ge_method == 'node2vec':
            results.append([param['d'], param['l'], param['n'], param['q'], param['p'], acc, f1, auc])
            print('dim={},length={},num={},q={},p={},accuracy={}，f1-score={},auc={}'
                  .format(param['d'], param['l'], param['n'], param['q'], param['p'], acc, f1, auc))
        elif ge_method in {'line', 'grafac'}:
            results.append([param['d'], acc, f1, auc])
            print('dim={},accuracy={}，f1-score={},auc={}'
                  .format(param['d'], acc, f1, auc))

    if ge_method == 'deepwalk':
        df = pd.DataFrame(data=results,
                          columns=['dim', 'length', 'num_walks', 'acc', 'f1', 'auc'])
    elif ge_method == 'node2vec':
        df = pd.DataFrame(data=results,
                          columns=['dim', 'length', 'num_walks', 'q', 'p', 'acc', 'f1', 'auc'])
    elif ge_method in {'line', 'grafac'}:
        df = pd.DataFrame(data=results,
                          columns=['dim', 'acc', 'f1', 'auc'])
    df.to_excel(result_file_path)


os.chdir('C:\\Users\\dell\\PycharmProjects\\riskgene')

# 文件路径
EMB_ROOT_PATH = 'data/emb/'
PPI_PATH = 'data/network/PPI-Network.txt'
CANDIDATE_NAME_PATH = 'data/candidate_genes.txt'
CANDIDATE_ID_PATH = 'data/candidate_id.txt'
DEEPWALK_EMB_ROOT_PATH = os.path.join(EMB_ROOT_PATH, 'deepwalk')
NODE2VEC_EMB_ROOT_PATH = os.path.join(EMB_ROOT_PATH, 'node2vec')
LINE_EMB_ROOT_PATH = os.path.join(EMB_ROOT_PATH, 'line')
grafac_emb_root_path = os.path.join(EMB_ROOT_PATH, 'grafac')

# 获取candidate gene的id
positive_gene_name_set, positive_gene_id_set = ppi.get_candidate_gene(CANDIDATE_NAME_PATH, CANDIDATE_ID_PATH)
# training(emb_path=LINE_EMB_ROOT_PATH, cl_method='nb', ge_method='line')
# training(emb_path=DEEPWALK_EMB_ROOT_PATH, cl_method='nb', ge_method='deepwalk')
# training(emb_path=NODE2VEC_EMB_ROOT_PATH, cl_method='nb', ge_method='node2vec')
# training(emb_path=grafac_emb_root_path, cl_method='nb', ge_method='grafac')
