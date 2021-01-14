from Function import F
from Newton import Newton_with_backtracking
import random
import numpy as np
import matplotlib.pyplot as plt

class Barrier_SVM_Classifier:
    def __init__(self, trainx, trainy, testx, testy, algo_type, kernel_type, margin_type, param, tol, C, rho, c, mu, t0):
        self.x = trainx
        self.y = trainy
        self.testx = testx
        self.testy = testy
        self.algo_type = algo_type
        self.kernel_type = kernel_type
        self.margin_type = margin_type
        self.param = param
        self.tol = tol
        self.C = C
        self.rho = rho
        self.c = c
        self.mu = mu
        self.t = t0
        self.accuracy()
    
    def train(self):
        f = F(self.kernel_type, self.margin_type, self.C, self.param, self.x, self.y, 1e4)
        A = self.y.reshape(-1,1) # Linear constraint matrix
        newton = Newton_with_backtracking(f, A.T, self.c, self.rho, self.tol)
        t = self.t
        fplot = []
        while (len(self.y)/t) > 1e-8:
            alpha_star = newton.descent_line_search()
            t = self.mu*t
            f.update_alpha(alpha_star)
            f.update_t(t)
            newton.update_F(f)
            print(f.f)
            fplot.append(f.f.item())
        plt.plot(fplot)
        plt.xlabel('Iterations')
        plt.ylabel('-W(\u03B1)')
        plt.title(self.margin_type+" Margin SVM using "+self.algo_type+" with "+self.kernel_type+" Kernel")
        plt.show()
        return f.alpha
    
    def accuracy(self):
        alpha_star = self.train()
        b_star = self.calc_b(alpha_star)
        y_pred = self.predict(alpha_star, b_star, self.testx, self.testy)
        accuracy = np.sum(y_pred==self.testy)/ y_pred.size
        print("Test Accuracy of", self.margin_type, "SVM with", self.kernel_type, "is", accuracy)

    def calc_b(self,alpha):
        bound = 100 #min(alpha)+2*np.std(alpha)
        sup_vec_idx = np.argwhere(alpha < bound).T[0] # find support vectors
        #print(len(sup_vec_idx))
        pred = np.zeros(len(sup_vec_idx))
        idx = 0
        for i in sup_vec_idx:
            for j in range(len(self.y)):
                pred[idx] += alpha[j]*self.y[j]*self.calc_ker(self.x[j],self.x[i])
            idx+=1
        b = self.y[sup_vec_idx] - pred
        return np.sum(b) / len(pred)
                
    
    def calc_ker(self, x1, x2):
        if self.kernel_type == "Gaussian":
            ker = np.exp(-self.param*np.linalg.norm(x1-x2)**2)
        if self.kernel_type == "Polynomial":
            ker = np.dot(x1,x2)**self.param
        return ker
    
    def predict(self, alpha, b, testx, testy):
        n = len(testy)
        m = len(self.y)
        pred = np.zeros(n)
        for i in range(n):
            for j in range(m):
                pred[i] += alpha[j]*self.y[j]*self.calc_ker(self.x[j],testx[i])
            pred[i] = np.sign(pred[i]+b)
        return pred

    
class SMO_SVM_Classifier:
    def __init__(self, C, tol, max_passes, trainx, trainy, testx, testy, algo_type, kernel_type, margin_type, param):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.x = trainx
        self.y = trainy
        self.testx = testx
        self.testy = testy
        self.F = F(kernel_type, margin_type, C, param, trainx, trainy, 1e4)
        self.algo_type = algo_type
        self.kernel_type = kernel_type
        self.margin_type = margin_type
        self.param = param
        
        # Initialize
        self.alpha = np.zeros(len(self.y)) #self.initialize_alpha() #np.zeros(len(self.y))
        self.b = 0

        # Execute
        self.train()
        self.accuracy()
        
    def train(self):
        passes = 0
        m = len(self.y)
        fplot = []
        f = F(self.kernel_type, self.margin_type, self.C, self.param, self.x, self.y, 1e4)
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                Ei = self.calc_E(i)
                if ((self.y[i]*Ei < -self.tol) and (self.alpha[i]<self.C)) or ((self.y[i]*Ei > self.tol) and (self.alpha[i]>0)):
                    
                    print('Iter {}: f* = {}'.format(i, self.F.f0(self.alpha[np.newaxis].T)))
                    fplot.append(f.f0(self.alpha[np.newaxis].T).item())
                    
                    j = random.choice([idx for idx in range(0, m) if idx != i])
                    Ej = self.calc_E(j)
                    
                    # save old alphas
                    ai_old = self.alpha[i].copy()
                    aj_old = self.alpha[j].copy()
                    
                    #compute L and H
                    if self.y[i]==self.y[j]:
                        L = max(0,ai_old + aj_old-self.C)
                        H = min(self.C, ai_old+aj_old)
                    else:
                        L = max(0,aj_old-ai_old)
                        H = min(self.C, self.C+ai_old-aj_old)
                    #end if
                    
                    if (L == H):
                        continue
                    
                    #compute eta
                    eta = (2*self.calc_ker(i,j))-self.calc_ker(i,i)-self.calc_ker(j,j)
                    
                    if (eta>=0):
                        continue
                        
                    # compute aj
                    unclipped_j = self.alpha[j] - ((self.y[j]*(Ei-Ej))/eta)
                    #clip j
                    if unclipped_j>H:
                        self.alpha[j] = H
                    elif (L<=unclipped_j<=H):
                        self.alpha[j] = unclipped_j
                    else:
                        self.alpha[j] = L
                    
                    
                    if abs(self.alpha[j]-aj_old)<1e-5:
                        continue
                    
                    # compute ai
                    self.alpha[i] = self.alpha[i]+(self.y[i]*self.y[j]*(aj_old-self.alpha[j]))
                    
                    # compute b
                    b1 = self.b - Ei - (self.y[i]*(self.alpha[i]-ai_old)*self.calc_ker(i,i)) - (self.y[j]*(self.alpha[j]-aj_old)*self.calc_ker(i,j))
                    b2 = self.b - Ej - (self.y[i]*(self.alpha[i]-ai_old)*self.calc_ker(i,j)) - (self.y[j]*(self.alpha[j]-aj_old)*self.calc_ker(j,j))
                    
                    if 0<self.alpha[i]<self.C:
                        self.b = b1
                    elif 0<self.alpha[j]<self.C:
                        self.b = b2
                    else:
                        self.b = (b1+b2)/2
                    
                    num_changed_alphas+=1
                    
                #end if
            #end for
            
            if num_changed_alphas == 0:
                passes+=1
            else:
                passes=0                      
        
        
        plt.plot(fplot)
        plt.xlabel('Iterations')
        plt.ylabel('-W(\u03B1)')
        plt.title(self.margin_type+" Margin SVM using "+self.algo_type+" with "+self.kernel_type+" Kernel")
        plt.show()
        
    def calc_ker(self,idx_i, idx_j):
        x1 = self.x[idx_i]
        x2 = self.x[idx_j]     
        if self.kernel_type == "Gaussian":
            ker = np.exp(-self.param*np.linalg.norm(x1-x2)**2)
        if self.kernel_type == "Polynomial":
            ker = np.dot(x1,x2)**self.param
        return ker
    
    def calc_ker2(self,x1, x2):
        if self.kernel_type == "Gaussian":
            ker = np.exp(-self.param*np.linalg.norm(x1-x2)**2)
        if self.kernel_type == "Polynomial":
            ker = np.dot(x1,x2)**self.param
        return ker
    
    def predict(self,x):
        pred = np.zeros(len(self.x))
        for i in range(len(self.x)):
            pred[i] = self.alpha[i]*self.y[i]*self.calc_ker2(self.x[i],x)
        return np.sign(np.sum(pred)+self.b)
    
    def calc_E(self,idx):
        x = self.x[idx]
        pred = self.predict(x)
        E = pred - self.y[idx]
        return E
                                                   
    def accuracy(self):
        n = len(self.testy)
        m = len(self.y)
        y_pred = np.zeros(n)
        for i in range(n):
            for j in range(m):
                y_pred[i]+= self.alpha[j]*self.y[j]*self.calc_ker2(self.x[j],self.testx[i])
            y_pred[i] = np.sign(y_pred[i]+self.b)
        acc = np.sum(y_pred==self.testy)/ n
        print("Test Accuracy of", self.margin_type, "SVM with", self.kernel_type, "is", acc)
        
    def initialize_alpha(self):
        alpha_0 = ((self.y==-1) * np.count_nonzero(self.y==1) + (self.y==1) * np.count_nonzero(self.y==-1))/10000000
        #alpha_0 = alpha_0[np.newaxis]
        return alpha_0 #alpha_0.T