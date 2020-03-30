import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import pandas as pd
def softthres(x,e):
    a=np.maximum(x-e,0)
    b=np.maximum(-1*x-e,0)
    return a-b



# base+sparsity+correlation
def base_sparsity_corrleation(X,Y,alpha,beta,enta,maxIter,miniLossMargin):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta:label correlation parameter
    enta:initiave w
    maxIter: max interation
    """
    n_samples, n_features = X.shape
    # X.T *X
    XTX=np.dot(np.transpose(X), X)
    # X.T*Y
    XTY=np.dot(np.transpose(X), Y)
    #初始化wo,w1
    W_s = np.dot(np.linalg.inv(XTX + enta * np.eye(n_features)),XTY).astype(np.float)
    W_s_1 = W_s
    # 计算相似度距离
    H =pairwise_distances(np.transpose(Y),metric="cosine")
    # L = cosine_similarity(np.transpose(Y))
    # R=np.diag(L.sum(1))-L
    iter = 1
    oldloss = 0
    df = pd.DataFrame(columns=["step", "loss"])
    # 计算Lipschitz constant
    Lip=math.sqrt(2*math.pow((np.linalg.norm(XTX,ord=2)),2)+math.pow(np.linalg.norm(beta*H ,ord=2),2))
    # 初始化b0,b1
    bk=1
    bk_1=1
    # accelerate proximal gradient
    while iter<=maxIter:
        W_s_k = W_s + np.dot((bk_1 - 1) / bk ,(W_s - W_s_1)).astype(np.float)
        Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) + beta * np.dot(W_s_k , H))
        bk_1 = bk
        bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
        W_s_1 = W_s
        # soft-thresholding operation
        W_s = softthres(Gw_s_k, alpha / Lip)

        a=np.transpose(np.dot(X,W_s)-Y)
        b=np.dot(X,W_s)-Y
        # 计算最小二乘损失
        predictionLoss=np.trace(np.dot(a,b))
        # 计算相关性
        correlation=np.trace(np.dot(H,np.dot(np.transpose(W_s),W_s)))
        # 计算稀疏
        sparsity=np.sum(np.sum(np.int64(W_s!=0),axis=0),axis=1)

        # 整体损失
        totalloss = predictionLoss + beta * correlation + alpha * sparsity

        df = df.append(pd.DataFrame({'step': [iter], 'loss': [math.fabs(oldloss - totalloss)]}))

        if math.fabs(oldloss-totalloss)<=miniLossMargin:
            break
        elif totalloss<=0:
            break
        else:
            oldloss = totalloss

        iter = iter + 1

    df.to_excel("stacking_iter_loss.xls")

    return W_s


# base
def base(X,Y,alpha,beta,enta,maxIter,miniLossMargin):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta:label correlation parameter
    enta:initiave w
    maxIter: max interation
    """
    n_samples, n_features = X.shape
    # X.T *X
    XTX=np.dot(np.transpose(X), X)
    # X.T*Y
    XTY=np.dot(np.transpose(X), Y)

    W_s = np.dot(np.linalg.inv(XTX),XTY).astype(np.float)
    iter = 1
    oldloss = 0
    df = pd.DataFrame(columns=["step", "loss"])

    while iter<=maxIter:

        a=np.transpose(np.dot(X,W_s)-Y)
        b=np.dot(X,W_s)-Y
        # 计算最小二乘损失
        predictionLoss=np.trace(np.dot(a,b))

        totalloss = predictionLoss

        df = df.append(pd.DataFrame({'step': [iter], 'loss': [math.fabs(oldloss - totalloss)]}))

        if math.fabs(oldloss-totalloss)<=miniLossMargin:
            break
        elif totalloss<=0:
            break
        else:
            oldloss = totalloss

        iter = iter + 1

    df.to_excel("stacking_iter_loss.xls")

    return W_s

# base+sparsity
def base_sparsity(X,Y,alpha,beta,enta,maxIter,miniLossMargin):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta:label correlation parameter
    enta:initiave w
    maxIter: max interation
    """
    n_samples, n_features = X.shape
    # X.T *X
    XTX=np.dot(np.transpose(X), X)
    # X.T*Y
    XTY=np.dot(np.transpose(X), Y)
    #初始化wo,w1
    W_s = np.dot(np.linalg.inv(XTX + enta * np.eye(n_features)),XTY).astype(np.float)
    W_s_1 = W_s

    iter = 1
    oldloss = 0
    df = pd.DataFrame(columns=["step", "loss"])
    # 计算Lipschitz constant
    Lip=math.sqrt(2*math.pow((np.linalg.norm(XTX,ord=2)),2))
    # 初始化b0,b1
    bk=1
    bk_1=1
    # accelerate proximal gradient
    while iter<=maxIter:
        W_s_k = W_s + np.dot((bk_1 - 1) / bk ,(W_s - W_s_1)).astype(np.float)
        Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) )
        bk_1 = bk
        bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
        W_s_1 = W_s
        # soft-thresholding operation
        W_s = softthres(Gw_s_k, alpha / Lip)

        a=np.transpose(np.dot(X,W_s)-Y)
        b=np.dot(X,W_s)-Y
        # 计算最小二乘损失
        predictionLoss=np.trace(np.dot(a,b))

        # 计算稀疏
        sparsity=np.sum(np.sum(np.int64(W_s!=0),axis=0),axis=1)

        # 整体损失
        totalloss = predictionLoss + alpha * sparsity

        df = df.append(pd.DataFrame({'step': [iter], 'loss': [math.fabs(oldloss - totalloss)]}))

        if math.fabs(oldloss-totalloss)<=miniLossMargin:
            break
        elif totalloss<=0:
            break
        else:
            oldloss = totalloss

        iter = iter + 1

    df.to_excel("stacking_iter_loss.xls")

    return W_s