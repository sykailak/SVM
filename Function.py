import random
import numpy as np

class F:
    def __init__(self, kernel_type, margin_type, C, param, trainx, trainy, t=1e4):
        self.kernel_type = kernel_type
        self.margin_type = margin_type
        self.param = param
        self.C = C
        self.t = t
        self.x = trainx
        self.y = trainy
        self.D = self.calc_D()
        self.alpha = self.initialize_alpha()
        self.assign_value()
        
    def f0(self, alpha):
        f0 = 0.5  * alpha.T @ self.D @ alpha - np.sum(alpha)
        return f0
    
    def _f(self, alpha):
        main = 0.5 * self.t * alpha.T @ self.D @ alpha - self.t * np.sum(alpha)
        if self.margin_type == "Hard":
            loss = - np.sum(np.log(alpha))
        else:
            loss = - np.sum(np.log(alpha)) - np.sum(np.log(self.C * np.ones_like(alpha) - alpha))
        return mainn + loss
        
    def _df(self, alpha):
        main = self.t * self.D @ alpha - self.t * np.ones_like(alpha)
        if self.margin_type == "Hard":
            loss = - 1/alpha
        else:
            loss = - 1/alpha - 1/(alpha-self.C * np.ones_like(alpha))
        return main + loss
        
    def _d2f(self, alpha):
        main = self.t * self.D
        if self.margin_type == "Hard":
            loss = np.diag((1/alpha**2).flatten())
        else:
            loss = np.diag((1/alpha**2).flatten()) + np.diag((1/(alpha-self.C)**2).flatten())
        return main + loss
    
    def calc_D(self):
        x_dim = self.x.shape[0]
        D = np.zeros((x_dim, x_dim))
        for i in range(x_dim):
            for j in range(x_dim):
                D[i,j] = self.y[i] * self.y[j] * self.calc_ker(self.x[i], self.x[j])
        return D
    
    def calc_ker(self,x1, x2):
        if self.kernel_type == "Gaussian":
            axis = None if x2.ndim==1 else 1
            ker = np.exp(-self.param*np.linalg.norm(x1-x2)**2)
        if self.kernel_type == "Polynomial":
            ker = np.dot(x1,x2)**self.param
        return ker
    
    def initialize_alpha(self):
        alpha_0 = ((self.y==-1) * np.count_nonzero(self.y==1) + (self.y==1) * np.count_nonzero(self.y==-1))/10000
        alpha_0 = alpha_0[np.newaxis]
        return alpha_0.T

    def update_alpha(self, alpha):
        self.alpha = alpha
        self.assign_value()
        
    def update_t(self,t):
        self.t = t
        self.assign_value()
        
    def assign_value(self):
        self.f = self._f(self.alpha)
        self.df = self._df(self.alpha)
        self.d2f = self._d2f(self.alpha)
        
