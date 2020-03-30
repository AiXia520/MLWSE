from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from read_arff import *
import sklearn.metrics as metrics
from mlmetrics import *
from lasso import *
from read_dataset import *
from read_matfile import *
import sys
sys.path.append("util")
import blockwise_descent, blockwise_descent_semisparse
import scipy
import time
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.ext import Meka, download_meka
from skmultilearn.model_selection import IterativeStratification

# Calculate all results
def calculate_all(np_test, np_pred,pre_score, model_time,output,isPrint=False):
    value = list()
    value.append(accuracy(np_test,np_pred))
    value.append(precision(np_test,np_pred))
    value.append(recall(np_test,np_pred))
    value.append(fscore(np_test,np_pred))
    value.append(hamloss(np_test,np_pred))
    value.append(subset(np_test,np_pred))
    value.append(microfscore(np_test,np_pred))
    value.append(macrofscore(np_test,np_pred))
    value.append(metrics.label_ranking_loss(np_test, pre_score))
    value.append(model_time)
    output.append(value)
    if isPrint:
        print("---------------------------------")
        print("Accuracy : {0:.4f}".format(value[0]))
        print("Precision: {0:.4f}".format(value[1]))
        print("Recall   : {0:.4f}".format(value[2]))
        print("F1-Score : {0:.4f}".format(value[3]))
        print("HammingL : {0:.4f}".format(value[4]))
        print("Subset   : {0:.4f}".format(value[5]))
        print("Micro - F1-Score : {0:.4f}".format(value[6]))
        print("Macro - F1-Score : {0:.4f}".format(value[7]))
        print("Rankloss : {0:.4f}".format(value[8]))
        print("run_time : {0:.4f}".format(value[9]))
        print("----------------------------------")
    del value
    return(output)


# BR classifier
def BR(X_train,y_train,X_test,new_X_test):
    classifier = BinaryRelevance(
        classifier=SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    new_pro_predictions = classifier.predict_proba(new_X_test)
    return(predictions,pro_predictions,new_pro_predictions)

# CC classifier
def CC(X_train,y_train,X_test,new_X_test):
    classifier = ClassifierChain(
        classifier=SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    new_pro_predictions = classifier.predict_proba(new_X_test)
    return(predictions,pro_predictions,new_pro_predictions)

# LP classifier
def LP(X_train,y_train,X_test,new_X_test):
    classifier = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=20),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    new_pro_predictions = classifier.predict_proba(new_X_test)
    return(predictions,pro_predictions,new_pro_predictions)


# RAkEL classifier
def RAkEL(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.RAkEL",
        weka_classifier="weka.classifiers.trees.J48",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_201\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

# MLS classifier
def MLS(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.BR",
        weka_classifier="weka.classifiers.meta.Stacking",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_201\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

# Feature combine the output of each base classifier
def combine_feature(pro_br,pro_cc,pro_lp):
    br = pd.DataFrame(pro_br.todense())
    cc = pd.DataFrame(pro_cc.todense())
    lp = pd.DataFrame(pro_lp.todense())
    X_new = pd.concat([br, cc, lp], axis=1)
    return(X_new)


if __name__ == '__main__':

    # dataset = "emotions"
    # label_count = 6

    # # read arff datasets
    # dataset,label_count=read_dataset()
    # path = "data/" + dataset + "/" + dataset
    # # read arff  file data
    # X, Y = read_arff(path, label_count)

    # read mat data
    dataset = "simulation_data1.mat"
    # read mat file data
    X, Y = read_matfile(dataset)

    print("current datasets："+dataset)
    # 5-fold cross validation 
    temp_mean=list()
    temp_std=list()

    k_fold = IterativeStratification(n_splits=5, order=1)
    for train, test in k_fold.split(X, Y):
        output = list()
        j_fold = IterativeStratification(n_splits=2, order=1)
        for new_train, new_test in j_fold.split(X[train],Y[train]):
            start_time = time.time()
            predictions_BR, pro_predictions_BR,new_pro_predictions_BR = BR(X[train][new_train], Y[train][new_train],
                                                    X[train][new_test], X[test])
            predictions_CC, pro_predictions_CC,new_pro_predictions_CC= CC(X[train][new_train], Y[train][new_train],
                                                    X[train][new_test], X[test])
            predictions_LP, pro_predictions_LP,new_pro_predictions_LP = LP(X[train][new_train], Y[train][new_train],
                                                    X[train][new_test], X[test])
            stacking = combine_feature(pro_predictions_BR, pro_predictions_CC, pro_predictions_LP)
            # trianing w
            w = lasso(stacking.values, Y[train][new_test].todense(),
                      math.pow(10, -4),math.pow(10, -3), 0.1, 200,0.0001)
            new_stacking=combine_feature(new_pro_predictions_BR, new_pro_predictions_CC, new_pro_predictions_LP)

            pre_score = np.dot(new_stacking.values, w)
            pred_label = np.rint(pre_score)
            pred_label = np.int64(pred_label >= 1)
            model_time=time.time() - start_time
            calculate_all(np.array(Y[test].todense()), np.array(pred_label), pre_score,model_time, output,False)

        data = pd.DataFrame(output)
        temp_mean.append(data.mean())
        temp_std.append(data.std())
        del output
    # The cross prediction results of each fold are obtained mean+/-std
    result_mean = pd.DataFrame(temp_mean)
    result_std = pd.DataFrame(temp_std)
    # output results
    result=pd.DataFrame({'Evaluate':['Accuracy','Precision','Recall','F1-Score','Hammingloss',
                                   'Subset','Micro-F1-Score','Macro-F1-Score','Rankloss','run_time'],
                         'Mean':result_mean.mean().values,'Std':result_std.mean().values})
    print(result)

