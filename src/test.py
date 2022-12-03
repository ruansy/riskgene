# 测试sample weight设置问题

import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src import classifiers as cl
from src import ppi


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
    sample_weights1 = [risklevel[int(line[0])] * class_weight[1] if int(line[0]) in keys else class_weight[0] for line
                       in
                       data]
    sample_weights2 = [class_weight[1] if int(line[0]) in keys else class_weight[0] for line in
                       data]
    sample_weights3 = [1 for line in data]

    return dim, X, target, sample_weights1, sample_weights2, sample_weights3


def line_training(positive_gene_id_set, risklevel: dict):
    line_filenames = os.listdir(LINE_EMB_ROOT_PATH)
    results = []
    for name in line_filenames:
        dim, X, y, weight_sample, weight_class, weight_one = parse_line_emb(name, positive_gene_id_set, risklevel)
        if dim != 512:
            continue
        x_train, x_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(X, y,
                                                                                                       weight_sample,
                                                                                                       test_size=0.3)
        acc, precision, recall, f1, auc, report = cl.nb(x_train, x_test, y_train, y_test, sample_weights_train)
        results = [acc, precision, recall, f1, auc, report]
        print("setting sample weight:")
        print(results)
        print(report)
        print('-------------')

        x_train, x_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(X, y,
                                                                                                       weight_class,
                                                                                                       test_size=0.3)
        acc, precision, recall, f1, auc, report = cl.nb(x_train, x_test, y_train, y_test, sample_weights_train)
        results = [acc, precision, recall, f1, auc, report]
        print("setting class weight:")
        print(results)
        print(report)
        print('-------------')

        x_train, x_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(X, y,
                                                                                                       weight_one,
                                                                                                       test_size=0.3)
        acc, precision, recall, f1, auc, report = cl.nb(x_train, x_test, y_train, y_test, sample_weights_train)
        results = [acc, precision, recall, f1, auc, report]
        print("not setting weight:")
        print(results)
        print(report)
        print('-------------')
    return results


def search_pn(k):
    positive_gene_id_set, risk_level = ppi.set_candidate_gene(GENECOUNT_PATH, k)
    print('k={}'.format(k))
    line_training(positive_gene_id_set, risk_level)
    return


def get_project_rootpath():
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

# search_pn(1)
search_pn(49)
