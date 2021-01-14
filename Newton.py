import random
import numpy as np

%

class Newton_with_backtracking:
    def __init__(self, F, A, c, rho, tol):
        self.F = F
        self.A = A
        self.c = c
        self.rho = rho
        self.Newton_step = self.Calc_Newton_step()
        self.tol = tol
        
    def Calc_Newton_step(self):
        K = np.hstack((self.F.d2f, self.A.T))
        K = np.vstack((K, np.hstack((self.A, [[0]]))))
        
        v = - np.vstack((self.F.df, 0))
        
        sol = np.linalg.inv(K) @ v
        newton_step = sol[:-1]
        
        return newton_step
    
    def backtracking(self):
        t = 1
        c = self.c
        rho = self.rho
        n = 1
        maxIter = 1e4
        stop = False
        p = self.Newton_step
        F = self.F
        tol = self.tol
        
        if F._f(F.alpha + t * p) <= F.f + c * t * F._df(F.alpha).T @ p:
            t = 1
        else:
            while n < maxIter and stop == False:
                t = rho * t
                if F._f(F.alpha + t * p) <= F.f + c * t * F._df(F.alpha).T @ p:
                    stop = True
                n += 1
        return t
    
    def descent_line_search(self):
        while np.linalg.norm(self.Newton_step) > self.tol * (1 + self.tol * np.abs(self.F.f)):
            t = self.backtracking()
            self.Newton_step = self.Calc_Newton_step()
            self.F.update_alpha(self.F.alpha + t * self.Newton_step)    
        return self.F.alpha
    
    def update_F(self, F):
        self.F = F
        