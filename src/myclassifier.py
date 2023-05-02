from sklearn import metrics, svm
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class MyClassifier():
    def __init__(self):
        self.classifiers = {
            'svm': svm.SVC(probability=True, C=1.0, kernel='rbf', gamma=1.0, cache_size=8192),
            'rf': RandomForestClassifier(n_jobs=-1, max_depth=40, n_estimators=20),
            'nb': GaussianNB()
        }

    def train(self, X, y, weight):
        for clf_name, clf in self.classifiers.items():
            clf.fit(X, y, weight)

    def predict(self, X):
        return {clf_name: clf.predict(X) for clf_name, clf in self.classifiers.items()}

    def predict_proba(self, X):
        return {clf_name: clf.predict_proba(X)[:, 1] for clf_name, clf in self.classifiers.items()}


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
