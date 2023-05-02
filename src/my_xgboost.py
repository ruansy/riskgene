import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.ppi import PPI

rcParams['figure.figsize'] = 12, 4


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


def dataloader(file_name, positive_gene: set, risklevel: dict):
    # file_name: emb files
    # positive_gene: 正例的gene
    # risklevel 出现了多少次的gene为positive gene

    data = file_name.strip('.emb').split('_')
    param = {i[0]: i[1:] for i in data[1:]}

    # 训练特征和label
    file_path = os.path.join(DEEPWALK_EMB_ROOT_PATH, file_name)
    with open(file_path, 'r') as f:
        data = [line.strip().split() for line in f.readlines()[1:]]
    X = [line[1:] for line in data]
    target = [1 if int(line[0]) in positive_gene else 0 for line in data]
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=int)

    # 权重
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=target)
    sample_weights = [risklevel[int(line[0])] * class_weight[1]
                      if int(line[0]) in positive_gene else class_weight[0] for line in data]
    return param, X, target, sample_weights


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


os.chdir(get_project_rootpath())
EMB_ROOT_PATH = 'data/emb/'
PPI_PATH = 'data/network/PPI-Network.txt'
DEEPWALK_EMB_ROOT_PATH = os.path.join(EMB_ROOT_PATH, 'deepwalk')
GENECOUNT_PATH = 'data/genecount.xls'

ppi = PPI(ppi_network_path=PPI_PATH, gene_count_path=GENECOUNT_PATH, k=6)

dataset = os.listdir(DEEPWALK_EMB_ROOT_PATH)[0]
params, X, y, weight = dataloader(dataset, ppi.positive_id, ppi.risk_level)
X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y,
                                                                               weight,
                                                                               test_size=0.3,
                                                                               random_state=420)

xgb_1 = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

xgb_1.fit(X_train, y_train)
y_pred = xgb_1.predict(X_test)
y_score = xgb_1.predict_proba(X_test)[:, 1]

# metric = MyMetrics(y_test, y_pred, y_score)
# acc, f1, auc, aupr = metric.evaluate(y_test, y_pred, y_score)
