from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from pandas.core.frame import DataFrame
class ClassifierProcessing:
    def __init__(self, name: str, clf: BaseEstimator, num_folds: int, method: str):
        self._num_folds = num_folds
        self._name = name
        self._clf = clf
        self._method = method

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @num_folds.deleter
    def num_folds(self):
        del self._num_folds

    @property
    def name(self) -> str:
        return self._name

    @name.deleter
    def name(self):
        del self._name

    @property
    def clf(self) -> BaseEstimator:
        return self._clf

    @clf.deleter
    def clf(self):
        del self._clf

    @property
    def method(self) -> str:
        return self._method

    @method.deleter
    def method(self):
        del self._method

    @property
    def y_scores(self) -> list:
        return self._y_scores

    @y_scores.deleter
    def y_scores(self):
        del self._y_scores

    @property
    def accuracy(self) -> list:
        return self._accuracy

    @accuracy.deleter
    def accuracy(self):
        del self._accuracy

    @property
    def precision(self) -> list:
        return self._precision

    @precision.deleter
    def precision(self):
        del self._precision

    @property
    def recall(self) -> list:
        return self._recall

    @recall.deleter
    def recall(self):
        del self._recall

    @property
    def f1(self) -> list:
        return self._f1

    @recall.deleter
    def f1(self):
        del self._f1

    @property
    def clf_precision(self) -> list:
        return self._clf_precision

    @clf_precision.deleter
    def clf_precision(self):
        del self._clf_precision

    
    @property
    def clf_recall(self) -> list:
        return self._clf_recall

    @clf_recall.deleter
    def clf_recall(self):
        del self._clf_recall

    
    @property
    def clf_thresholds(self) -> list:
        return self._clf_thresholds

    @clf_thresholds.deleter
    def clf_thresholds(self):
        del self._clf_thresholds

    @property
    def roc_tpr(self) -> list:
        return self._roc_tpr
    
    @roc_tpr.deleter
    def roc_tpr(self):
        del self._roc_tpr

    @property
    def roc_fpr(self) -> list:
        return self._roc_fpr
    
    @roc_fpr.deleter
    def roc_fpr(self):
        del self._roc_fpr

    @property
    def roc_threshold(self) -> list:
        return self._roc_threshold
    
    @roc_threshold.deleter
    def roc_threshold(self):
        del self.roc_threshold

    @property
    def roc_score(self) -> float:
        return self._roc_score
    
    @roc_score.deleter
    def roc_score(self):
        del self._roc_score

    def process(self, X: DataFrame, y: DataFrame):
        self._y_scores = self.get_predictions(X, y)
        self._accuracy = self.get_accuracy(X, y)
        self._precision = self.get_precision(X, y)
        self._recall = self.get_recall(X, y)
        self._f1 = self.get_f1(X, y)
        self.set_precision_recall_threshold(y)
        self.set_roc_values(y, self._y_scores)
        self._roc_score = self.get_roc_score(y, self._y_scores)

    def get_predictions(self, X: DataFrame, y: DataFrame) ->list:
        if(self._method == 'predict_proba'):
            y_probas = cross_val_predict(self._clf, X, y, cv=self._num_folds, method=self._method)
            return y_probas[:,1]
        elif(self._method == 'decision_function'):
            return cross_val_predict(self._clf, X, y, cv=self._num_folds, method=self._method)
        
    def get_accuracy(self, X: DataFrame, y: DataFrame) -> list:
        return cross_val_score(self._clf, X, y, cv=self._num_folds, scoring='accuracy')

    def get_precision(self, X: DataFrame, y: DataFrame) ->list:
        return cross_val_score(self._clf, X, y, cv=self._num_folds, scoring='precision')

    def get_recall(self, X: DataFrame, y: DataFrame) -> list:
        return cross_val_score(self._clf, X, y, cv=self._num_folds, scoring='recall')
        
    def get_f1(self, X :DataFrame, y :DataFrame) -> list:
        return cross_val_score(self._clf, X, y, cv=self._num_folds, scoring='f1')

    def set_precision_recall_threshold(self, y: DataFrame):
        precision, recall, thresholds = precision_recall_curve(y, self._y_scores)
        self._clf_precision = precision
        self._clf_recall = recall
        self._clf_thresholds = thresholds

    def set_roc_values(self, y: DataFrame, y_pred: DataFrame):
        self._roc_fpr, self._roc_tpr, self._roc_threshold = roc_curve(y, y_pred)

    def get_roc_score(self, y: DataFrame, y_pred: DataFrame):
        return roc_auc_score(y,y_pred)