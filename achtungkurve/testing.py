import os
import pickle

import numpy as np
from sklearn import tree
from sklearn.externals import joblib

from utils import ACTIONHOT

"""
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
"""

train = []
labels = []

for filename in os.listdir("baum/training"):
    with open("baum/training/" + filename, "rb") as fp:  # Unpickling
        data = pickle.load(fp)

        for cont in data:
            # cont = SaveState
            if not cont.result:
                continue

            board = np.asarray(cont.board).flatten()
            board = np.where(board > 0, 9, board)
            lab = ACTIONHOT[cont.action]

            train.append(board)
            labels.append(lab)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, labels)

joblib.dump(clf, 'baum/dt_3.pkl')

# res = clf.predict([train[2]])
# print(res, labels[2])
