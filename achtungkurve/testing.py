from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn import tree
import pickle
import numpy as np
from utils import SaveState, ACTIONHOT
"""
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
"""

train = []
labels = []

with open("baum/data.txt", "rb") as fp:  # Unpickling
    data = pickle.load(fp)

    for cont in data:
        # cont = SaveState
        if not cont.result:
            continue

        board = np.asarray(cont.board).flatten()
        lab = ACTIONHOT[cont.action]

        train.append(board)
        labels.append(lab)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(train, labels)

joblib.dump(clf, 'baum/dt_2.pkl')

# res = clf.predict([train[2]])
# print(res, labels[2])
