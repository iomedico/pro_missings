import numpy as np

class MyStackingClassifier(object):
    def __init__(self, clf_tuples, weights):
        self.estimators = clf_tuples
        self.weights = weights
        self.classifiers = [c[1] for c in clf_tuples]

    def predict_proba(self, X):
        return sum([w*c.predict_proba(X) for c, w in list(zip(self.classifiers, self.weights))])

    def predict(self, X):
        p = self.predict_proba(X)[:,1]
        return np.round(p).astype(int)

    def fit(self, X, y):
        self.estimators = [(clf_name, clf.fit(X,y)) for clf_name, clf in self.estimators]
        self.classifiers = [c[1] for c in self.estimators]
        return self