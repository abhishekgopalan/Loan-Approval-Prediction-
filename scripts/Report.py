import numpy as np
import pandas as pd
class Report:
    def __init__(self, clfs, num_folds):
        self._clfs = clfs
        self._num_folds = num_folds

    def compare_accuracy(self):
        arrays = []
        for clf in self._clfs:
            a = np.append(clf.name, clf.accuracy)
            arrays.append(a)
        columns = self.get_column_names('accuracy')
        return pd.DataFrame(arrays, columns=columns)

    def compare_precision(self):
        arrays = []
        for clf in self._clfs:
            a = np.append(clf.name, clf.precision)
            arrays.append(a)
        columns = self.get_column_names('precision')
        return pd.DataFrame(arrays, columns=columns)

    def compare_recall(self):
        arrays = []
        for clf in self._clfs:
            a = np.append(clf.name, clf.recall)
            arrays.append(a)
        columns = self.get_column_names('recall')
        return pd.DataFrame(arrays, columns=columns)

    def compare_f1(self):
        arrays = []
        for clf in self._clfs:
            a = np.append(clf.name, clf.f1)
            arrays.append(a)
        columns = self.get_column_names('f1')
        return pd.DataFrame(arrays, columns=columns)

    def get_column_names(self, type):
        iteration_names = [f'{type}_iteration_{idx}' for idx in range(1,self._num_folds + 1)]
        return np.append('name',iteration_names)
            