from sklearn import metrics


class MyMetrics:
    def __init__(self, y_true, y_pred, y_score=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score

    def accuracy(self):
        return metrics.accuracy_score(self.y_true, self.y_pred)

    def f1(self):
        return metrics.f1_score(self.y_true, self.y_pred)

    def auc(self):
        return metrics.roc_auc_score(self.y_true, self.y_score)

    def aupr(self):
        return metrics.average_precision_score(self.y_true, self.y_score)

    def evaluate(self, y_true, y_pred, y_score):
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_score)
        aupr = metrics.average_precision_score(y_true, y_score)
        return acc, f1, auc, aupr
