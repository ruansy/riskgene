from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Cmodel():
    def __init__(self, features, labels, sample_weights):
        self.y_score = None
        self.y_pred = None
        self.features = features
        self.labels = labels
        self.model = None
        self.features_train, self.features_test, self.labels_train, self.labels_test, self.weights_train, self.weights_test = train_test_split(
            features, labels, sample_weights, test_size=0.3)

    def set_model(self, model):
        self.model = model

    def training(self):
        self.model.fit(self.features_train, self.labels_train, sample_weight=self.weights_train)
        self.y_pred = self.model.predict(self.features_test)
        self.y_score = self.model.predict_proba(self.features_test)[:, 1]
        return

    def evaluation(self):
        # 计算准确率
        acc = metrics.accuracy_score(y_pred=self.y_pred, y_true=self.labels_test)
        # 计算精确率和召回率
        precision = metrics.precision_score(y_true=self.labels_test, y_pred=self.y_pred)
        recall = metrics.recall_score(y_true=self.labels_test, y_pred=self.y_pred)
        # 计算F1分数和AUC
        f1 = metrics.f1_score(y_true=self.labels_test, y_pred=self.y_pred)
        auc = metrics.roc_auc_score(y_true=self.labels_test, y_score=self.y_score)
        # 混淆矩阵
        confusion_matrix = metrics.confusion_matrix(y_true=self.labels_test, y_pred=self.y_pred)
        # 结果报告
        return acc, precision, recall, f1, auc, confusion_matrix

    def classify_report(self):
        report = metrics.classification_report(y_true=self.labels_test, y_pred=self.y_pred, digits=4)
        return report


# def forest(x_train, x_test, y_train, y_test, sample_weights):
# 超参数搜索
# param = {"n_estimators": [20, 60, 100, 140, 180, 200],
#          "max_depth": [10, 15, 20, 25, 30]}
# 训练
# for i in param['max_depth']:
#     rf = RandomForestClassifier(n_jobs=-1, max_depth=i)
#     rf.fit(x_train, y_train, sample_weight=sample_weights)
#     y_pred = rf.predict(x_test)
#     y_score = rf.predict_proba(x_test)[:, 1]
#     acc, precision, recall, f1, auc = evaluation(y_true=y_test, y_score=y_score, y_pred=y_pred)
#     report = metrics.classification_report(y_true=y_test,
#                                            y_pred=y_pred,
#                                            digits=4)
#     print('-----------------------')
#     print(i)
#     print(acc, precision, recall, f1, auc)
#     print(report)
# return acc, precision, recall, f1, auc, report


# def nb(x_train, x_test, y_train, y_test, sample_weights):
#     model = GaussianNB()
#     model.fit(x_train, y_train, sample_weight=sample_weights)
#     y_pred = model.predict(x_test)
#     y_score = model.predict_proba(x_test)[:, 1]
#     acc, precision, recall, f1, auc = evaluation(y_true=y_test, y_score=y_score, y_pred=y_pred)
#     report = metrics.classification_report(y_true=y_test,
#                                            y_pred=y_pred,
#                                            digits=4)
#     return acc, precision, recall, f1, auc, report


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
