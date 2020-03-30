import scipy.io
import scipy
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from  mlmetrics import *
import sklearn.metrics as metrics
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def read_matfile(dataset):

    path = "data/matfile/" + dataset
    mat= scipy.io.loadmat(path)
    # print(mat.keys())
    X = scipy.sparse.lil_matrix(mat['data'])
    Y = scipy.sparse.lil_matrix(mat['target'])
    Y=np.transpose(Y)
    return (X,Y)

if __name__ == '__main__':
    dataset = "enron.mat"
    X,Y=read_matfile(dataset)
    a = np.array(Y.todense())
    print(np.int64(a != 0).sum(0))

    # X_train, y_train, X_test, y_test = iterative_train_test_split(X, Y, test_size=0.3)
    #
    # classifier = BinaryRelevance(
    #     classifier=SVC(probability=True),
    #     require_dense=[False, True]
    # )
    #
    # # train
    # classifier.fit(X_train, y_train)
    #
    # # predict
    # predictions = classifier.predict(X_test)
    # predictions = classifier.predict(X_test)
    # pro_predictions = classifier.predict_proba(X_test)
    # print(metrics.hamming_loss(y_test, predictions))
    i=1
    k_fold = IterativeStratification(n_splits=5, order=1)
    for train, test in k_fold.split(X, Y):

        print("fold"+str(i))
        j_fold = IterativeStratification(n_splits=2, order=1)
        for new_train, new_test in j_fold.split(X[train], Y[train]):
            classifier = BinaryRelevance(
                classifier = SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
                require_dense = [False, True]
            )
            print(X[train][new_train])
            print(Y[train][new_train])
            print(X[train][new_test])
            # train
            classifier.fit(X[train][new_train], Y[train][new_train])

            # predict
            predictions = classifier.predict(X[train][new_test])
            predictions = classifier.predict(X[train][new_test])
            pro_predictions=classifier.predict_proba(X[train][new_test])
            print(metrics.hamming_loss(Y[train][new_test], predictions))


            classifier1 = ClassifierChain(
                classifier= SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
                require_dense=[False, True]
            )
            # train
            classifier1.fit(X[train][new_train], Y[train][new_train])
            # predict
            predictions = classifier1.predict(X[train][new_test])
            predictions = classifier1.predict(X[train][new_test])
            pro_predictions=classifier1.predict_proba(X[train][new_test])
            print(metrics.hamming_loss(Y[train][new_test], predictions))

            classifier2 = ClassifierChain(
                classifier=RandomForestClassifier(n_estimators=20),
                require_dense=[False, True]
            )
            # train
            classifier2.fit(X[train][new_train], Y[train][new_train])
            # predict
            predictions = classifier2.predict(X[train][new_test])
            predictions = classifier2.predict(X[train][new_test])
            pro_predictions=classifier2.predict_proba(X[train][new_test])
            print(metrics.hamming_loss(Y[train][new_test], predictions))

        i=i+1