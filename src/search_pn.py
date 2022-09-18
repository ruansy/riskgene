# 寻找最优秀的正负例策略
# 使用line算法的emb向量

import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src import classifiers as cl
from src import ppi


def class_balance(labels):
    weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
    return weight


def parse_line_emb(file_name, positive_gene_id_set, risklevel):
    data = file_name.strip().split('_')
    dim = int(data[1].rstrip('.emb')[1:])
    file_path = os.path.join(LINE_EMB_ROOT_PATH, file_name)
    with open(file_path, 'r') as f:
        data = [line.strip().split() for line in f.readlines()[1:]]
    X = [line[1:] for line in data]

    target = [1 if int(line[0]) in positive_gene_id_set else 0 for line in data]
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=int)

    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=target)
    keys = risklevel.keys()
    sample_weights = [risklevel[int(line[0])] * class_weight[1] if int(line[0]) in keys else class_weight[0] for line in
                      data]
    # sample_weights = [1 for line in data]
    return dim, X, target, sample_weights


def line_training(positive_gene_id_set, risklevel: dict):
    line_filenames = os.listdir(LINE_EMB_ROOT_PATH)
    results = []
    for name in line_filenames:
        dim, X, y, sample_weights = parse_line_emb(name, positive_gene_id_set, risklevel)
        x_train, x_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(X, y,
                                                                                                       sample_weights,
                                                                                                       test_size=0.3)
        acc, precision, recall, f1, auc, report = cl.nb(x_train, x_test, y_train, y_test, sample_weights_train)
        results.append([dim, acc, precision, recall, f1, auc, report])
        print('dim={}, accuracy={}, precision={}, recall={}, f1-score={}, auc={}'
              .format(dim, acc, precision, recall, f1, auc))
        print(report)

    return results


def search_pn():
    results = {}
    params = {
        'k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    for k in params['k']:
        positive_gene_id_set, risk_level = ppi.set_candidate_gene(GENECOUNT_PATH, k)
        print('k={}'.format(k))
        results[k] = line_training(positive_gene_id_set, risk_level)
        print('--------------------------')
    return


def get_project_rootpath():
    """
    获取项目根目录。此函数的能力体现在，不论当前module被import到任何位置，都可以正确获取项目根目录
    :return:
    """
    path = os.path.realpath(os.curdir)
    while True:
        for subpath in os.listdir(path):
            # PyCharm项目中，'.idea'是必然存在的，且名称唯一
            if '.idea' in subpath:
                return path
        path = os.path.dirname(path)


os.chdir(get_project_rootpath())

# 文件路径
PPI_PATH = 'data/network/PPI-Network.txt'
CANDIDATE_NAME_PATH = 'data/candidate_genes.txt'
CANDIDATE_ID_PATH = 'data/candidate_id.txt'
GENECOUNT_PATH = 'data/genecount.xls'
EMB_ROOT_PATH = 'data/emb/'
LINE_EMB_ROOT_PATH = os.path.join(EMB_ROOT_PATH, 'line')

nx_ppi_network, ppi_id_list = ppi.get_ppi_network(PPI_PATH)

resutls = search_pn()
