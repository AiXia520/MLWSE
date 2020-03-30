from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from read_arff import *
import sklearn.metrics as metrics
from mlmetrics import *
from read_dataset import *
from read_matfile import *
import sys
sys.path.append("util")
sys.path.append("simulation")
from simulation_lasso import *
import blockwise_descent, blockwise_descent_semisparse
import scipy
import time
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.ext import Meka, download_meka
from skmultilearn.model_selection import IterativeStratification
# 计算所有结果
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
    # value.append(metrics.label_ranking_loss(np_test, pre_score))
    value.append(0)
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


# BR分类器
def BR(X_train,y_train,X_test,new_X_test):
    classifier = BinaryRelevance(
        classifier=SVC(probability=True,C=1.0, kernel='linear',gamma='auto'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    new_pro_predictions = classifier.predict_proba(new_X_test)
    return(predictions,pro_predictions,new_pro_predictions)

# CC分类器
def CC(X_train,y_train,X_test,new_X_test):
    classifier = ClassifierChain(
        classifier=SVC(probability=True,C=1.0, kernel='poly',gamma='auto'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    new_pro_predictions = classifier.predict_proba(new_X_test)
    return(predictions,pro_predictions,new_pro_predictions)

# LP 分类器
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


#MLKNN分类器
def MLKNN(X_train,y_train,X_test,y_test):
    classifier = MLkNN(k=5)
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    return(predictions,pro_predictions)

# RAkEL 分类器
def RAkEL(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.RAkEL",
        weka_classifier="weka.classifiers.trees.J48",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_201\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

# MLS 分类器
def MLS(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.BR",
        weka_classifier="weka.classifiers.meta.Stacking",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_201\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

# 各个基分类器输出结果，特征降维联合
def combine_feature(pro_br,pro_cc,pro_lp):
    br = pd.DataFrame(pro_br.todense())
    cc = pd.DataFrame(pro_cc.todense())
    lp = pd.DataFrame(pro_lp.todense())
    # mlknn=pd.DataFrame(pro_mlknn.todense())
    # x_se = pd.DataFrame(X_se)
    X_new = pd.concat([br, cc, lp], axis=1)
    return(X_new)


if __name__ == '__main__':

    # dataset = "emotions"
    # label_count = 6

    # # 读入arff数据
    # dataset,label_count=read_dataset()
    # path = "data/" + dataset + "/" + dataset
    # # 读入ARFF文件数据
    # X, Y = read_arff(path, label_count)

    # 读入mat数据
    dataset = "simulation_data6.mat"
    # 读入mat文件数据
    X, Y = read_matfile(dataset)

    print("当前数据集为："+dataset)

    output = list()
    # 划分30%测试集
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, Y, test_size=0.3)
    # 划分剩余的35%为训练集，35%为验证集
    X_train_train, y_train_train, X_test_validation, y_test_validation = iterative_train_test_split(X_train, y_train, test_size=0.5)

    start_time = time.time()
    predictions_BR, pro_predictions_BR, new_pro_predictions_BR = BR(X_train_train, y_train_train,
                                                                    X_test_validation, X_test)
    predictions_CC, pro_predictions_CC, new_pro_predictions_CC = CC(X_train_train, y_train_train,
                                                                    X_test_validation, X_test)
    predictions_LP, pro_predictions_LP, new_pro_predictions_LP = LP(X_train_train, y_train_train,
                                                                    X_test_validation, X_test)

    stacking = combine_feature(pro_predictions_BR, pro_predictions_CC, pro_predictions_LP)

    # BR
    print("****************BR*********")

    pred_label = new_pro_predictions_BR.todense()
    pred_label = np.int64(pred_label >= 0.5)
    pre_score=0
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)

    # CC
    print("****************CC*********")

    pred_label = new_pro_predictions_CC.todense()
    pred_label = np.int64(pred_label >= 0.5)
    pre_score=0
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)

    # LP
    print("****************LP*********")

    pred_label = new_pro_predictions_LP.todense()
    pred_label = np.int64(pred_label >= 0.5)
    pre_score=0
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)

    # base
    print("****************base*********")
    w = base(stacking.values, y_test_validation.todense(),
              math.pow(10, -4), math.pow(10, -3), 0.1, 200, 0.0001)
    new_stacking = combine_feature(new_pro_predictions_BR, new_pro_predictions_CC, new_pro_predictions_LP)

    print(w)
    pre_score = np.dot(new_stacking.values, w)
    pred_label = np.rint(pre_score)
    pred_label = np.int64(pred_label >= 1)
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)

    # base+sparsity
    print("****************base+sparsity*********")
    w = base_sparsity(stacking.values, y_test_validation.todense(),
              math.pow(10, -4), math.pow(10, -3), 0.1, 200, 0.0001)
    new_stacking = combine_feature(new_pro_predictions_BR, new_pro_predictions_CC, new_pro_predictions_LP)

    print(w)
    pre_score = np.dot(new_stacking.values, w)
    pred_label = np.rint(pre_score)
    pred_label = np.int64(pred_label >= 1)
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)

    # base+group_sparsity
    print("****************base_group_sparsity*********")
    # 学习w
    groups = np.arange(stacking.values.shape[1]) // y_test_validation.todense().shape[1]
    model = blockwise_descent.SGL(groups=groups, alpha=0.05, lbda=math.pow(10, -3),
                                  beta=math.pow(10, -2), enta=0.1)
    model.fit(stacking.values, np.array(y_test_validation.todense()))
    w = model.coef_
    print(w)
    # 预测
    pre_score = np.dot(new_stacking.values, w)
    pred_label = np.rint(pre_score)
    pred_label = np.int64(pred_label >= 0.5)
    model_time = time.time() - start_time
    calculate_all(np.array(y_test.todense()), np.array(pred_label), pre_score, model_time, output, True)