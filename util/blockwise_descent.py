import numpy
import blockwise_descent_semisparse

#use block coordinate descent to optimize MLWSE-L21


class SGL(blockwise_descent_semisparse.SGL):
    def __init__(self, groups, alpha, lbda, beta, enta, max_iter_outer=500, max_iter_inner=100, rtol=1e-6):
        self.ind_sparse = numpy.ones((len(groups), ))
        self.groups = numpy.array(groups)
        self.alpha = alpha
        self.lbda = lbda
        self.beta = beta
        self.enta = enta
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rtol = rtol
        self.coef_ = None
