from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def evaluation(y_pred, y_true, y_score):
    acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    return acc, precision, recall, f1, auc


def forest(x_train, x_test, y_train, y_test,sample_weights):
    rf = RandomForestClassifier()
    # 超参数搜索
    param = {"n_estimators": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
             "max_depth": [25, 35, 45, 55, 65, 75, 85, 95]}
    gc = GridSearchCV(rf, param_grid=param, cv=5)
    # 训练
    gc.fit(x_train, y_train,sample_weight=sample_weights)
    # 交叉验证网格搜索的结果
    print("在测试集上的准确率：", gc.score(x_test, y_test))
    print("在验证集上的准确率：", gc.best_score_)
    print("最好的模型参数：", gc.best_params_)
    print("最好的模型：", gc.best_estimator_)


def mysvm(x_train, x_test, y_train, y_test):
    model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    acc, f1, auc = evaluation(y_true=y_test, y_score=y_score, y_pred=y_pred)
    return acc, f1, auc


def nb(x_train, x_test, y_train, y_test, sample_weights):
    model = GaussianNB()
    model.fit(x_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    acc, precision, recall, f1, auc = evaluation(y_true=y_test, y_score=y_score, y_pred=y_pred)
    report = metrics.classification_report(y_true=y_test,
                                           y_pred=y_pred,
                                           digits=4)
    return acc, precision, recall, f1, auc, report


def bagging(x_train, y_train, x_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    model = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0,
                              bootstrap=True,
                              bootstrap_features=False, n_jobs=1, random_state=1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # acc = accuracy_score(y_pred, y_test)
    # print("bagging accuracy : %.4g" % accuracy_score(y_pred, y_test))
    # return acc


def adaboost(x_train, y_train, x_test, y_test):
    AB = AdaBoostClassifier(n_estimators=1000, learning_rate=1.0, algorithm='SAMME', random_state=None)
    AB.fit(x_train, y_train)
    predict_results = AB.predict(x_test)
    # print("adaboost accuracy : %.4g" % accuracy_score(predict_results, y_test))


def xgboost(x_train, y_train, x_test, y_test):
    xgboost = XGBClassifier(learning_rate=0.01,
                            n_estimators=10,  # 树的个数-10棵树建立xgboost
                            max_depth=4,  # 树的深度
                            min_child_weight=1,  # 叶子节点最小权重
                            gamma=0.,  # 惩罚项中叶子结点个数前的参数
                            subsample=1,  # 所有样本建立决策树
                            colsample_btree=1,  # 所有特征建立决策树
                            scale_pos_weight=1,  # 解决样本个数不平衡的问题
                            random_state=27,  # 随机数
                            slient=0
                            )
    xgboost.fit(x_train, y_train)
    y_pred = xgboost.predict(x_test)
    # print("xgboost accuracy : %.4g" % accuracy_score(y_test, y_pred))

# class Classifier():
#     def __init__(self, data, candidate, risklevel):
#         self.data = data
#         self.candidate = candidate
#         self.risklevel = risklevel
#         self.X = None
#         self.y = None
#
#     def generate_training_data(self):
#         X = [line[1:] for line in self.data]
#
#         target = [1 if int(line[0]) in self.candidate else 0 for line in self.data]
#         self.X = np.asarray(X, dtype=float)
#         target = np.asarray(target, dtype=int)
#         class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=target)
#         keys = self.risklevel.keys()
#         self.sample_weights = [
#             self.risklevel[int(line[0])] * class_weight[1] if int(line[0]) in keys else class_weight[0]
#             for
#             line in
#             self.data]
#         return

# def classify(method, data, candidate, risklevel):
#     # methods = ('svm', 'rf', 'nb')
#     X = [line[1:] for line in data]
#     target = [1 if int(line[0]) in candidate else 0 for line in data]
#     X = np.asarray(X, dtype=float)
#     target = np.asarray(target, dtype=int)
#     class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=target)
#     keys = risklevel.keys()
#     sample_weights = [
#         risklevel[int(line[0])] * class_weight[1] if int(line[0]) in keys else class_weight[0]
#         for line in data]
#     if method=='nb':
#         model = GaussianNB()
#     elif method=='svm':
#         model=svm.SVC()
