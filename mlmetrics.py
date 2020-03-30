import numpy as np 
from scipy.sparse import issparse
def pre_cal(y_true, y_pred, print_all = False):
    if(y_true.shape != y_pred.shape):
        print("Wrong y_preds matrics!")

    real_pos = real_neg = pred_pos = pred_neg  = true_pos = true_neg = []

    for i in range(y_true.shape[0]):
        # real values - RP and RN
        real_pos = np.asarray(np.append(real_pos,np.logical_and(y_true[i], y_true[i]).sum()), dtype=np.int64).reshape(-1,1)
        real_neg = np.asarray(np.append(real_neg,np.logical_and(np.logical_not(y_true[i]),np.logical_not(y_true[i])).sum()), dtype=np.int64).reshape(-1,1)

        # y_pred values - PP and PN
        pred_pos = np.asarray(np.append(pred_pos,np.logical_and(y_pred[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
        pred_neg = np.asarray(np.append(pred_neg,np.logical_and(np.logical_not(y_pred[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)

        # true labels - TP and TN
        true_pos = np.asarray(np.append(true_pos,np.logical_and(y_true[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
        true_neg = np.asarray(np.append(true_neg,np.logical_and(np.logical_not(y_true[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)

    if print_all:
		# if print_all = True - it prints RP, RN, PP, PN, TP and TN
        result = np.concatenate((real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg), axis=1)
        # print(result)

    return(real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg)

#function to resolve divide by zero error and accept the value 0 when divided by 0
def divideZero( value_a, value_b):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide( value_a, value_b )
        result[ ~ np.isfinite( result )] = 0
    return result

def accuracy(y_true, y_pred):
    #return the accuracy - example based
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = (true_pos + true_neg)/(pred_pos + pred_neg)
    score = np.mean(score)
    return score


def precision(y_true, y_pred):
    #return precision - example based
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = divideZero(true_pos, pred_pos)
    score = np.mean(score)
    return score

def recall(y_true, y_pred):
    #return precision - example based
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = divideZero(true_pos, real_pos)
    score = np.mean(score)
    return score


def fscore(y_true, y_pred,beta = 1):
	#return f(beta)score - example based : default beta value is 1
    prec, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    beta_val = beta*beta
    score = ((1+beta_val)*(prec*rec))/(beta_val*(prec+rec))
    return score


def hamloss(y_true, y_pred):
	#return hamming loss - example based
    hamloss = list()
    for i in range(y_true.shape[0]):
        hamloss = np.asarray(np.append(hamloss,np.logical_xor(y_true[i], y_pred[i]).sum()), dtype=np.int64).reshape(-1,1)
    score = (hamloss.sum())/((y_true.shape[0])*(y_true.shape[1]))
    return score


def subset(y_true, y_pred):
	#return subset accuracy - example based
    subset_matrix = list()
    for i in range(y_true.shape[0]):
        subset_matrix = np.asarray(np.append(subset_matrix, np.array_equal(y_true[i],y_pred[i])), dtype=np.int64).reshape(-1,1)
    score = (subset_matrix.sum())/((y_true.shape[0])*(y_true.shape[1]))
    return score

def zeroloss(y_true, y_pred):
    #return new array with removed element having all zero in y_true
    condition = list()
    index = list()
    for i in range(y_true.shape[0]):
        new_true = new_pred = list()
        condition = np.logical_and(y_true[i],y_true[i]).sum()
        if (condition==0):
            index = np.asarray(np.append(index,i), dtype = np.int64)

        new_true = np.delete(y_true,index, axis = 0)
        new_pred = np.delete(y_pred,index, axis = 0)
    return new_true, new_pred

def microprecision(y_true, y_pred):
    #return micro-precision
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = true_pos.sum()/pred_pos.sum()
    return score

def microrecall(y_true, y_pred):
    #return micro-recall
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = true_pos.sum()/real_pos.sum()
    return score

def microfscore(y_true, y_pred,beta = 1):
    #return micro-fscore
    prec, rec = microprecision(y_true, y_pred), microrecall(y_true, y_pred)
    beta_val = beta*beta
    score = ((1+beta_val)*(prec*rec))/(beta_val*(prec+rec))
    return score

def macroprecision(y_true, y_pred):
    #return macro-precision
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = divideZero(true_pos, pred_pos)
    return score

def macrorecall(y_true, y_pred):
    #return macro-recall
    real_pos, real_neg, pred_pos, pred_neg, true_pos, true_neg = pre_cal(y_true,y_pred)
    score = divideZero(true_pos, real_pos)
    return score

def macrofscore(y_true, y_pred,beta = 1):
    #return macro-fscore
    prec, rec = macroprecision(y_true, y_pred), macrorecall(y_true, y_pred)
    beta_val = beta*beta
    score = divideZero(((1+beta_val)*(prec*rec)),(beta_val*(prec+rec)))
    score = np.mean(score)
    return score


def precision_at_ks(true_Y, pred_Y, ks=[1, 2, 3, 4, 5]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / k
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = np.mean(precs)
    return result


def print_hdf5_object(o):
    for k in o:
        print('{}: {}'.format(k, o[k].value))

