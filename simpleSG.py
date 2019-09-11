'''
simpleSG.py --- simpleStochasticGradient

The step function of all the optimizers takes the gradient vector 
as an argument. The only required initialization parameter for all 
optimizers is "eta_init" which is the starting point (initial "guess"). 

Note: All gradient steps are in the ascent direction (maximization)

Rob Salomone (September 10th 2019)
 '''
import numpy as np

class SGD():
    ''' SGD with Momentum '''
    def __init__(self, eta_init, rho = 0.9, lr = 1e-5):
        self.eta = eta_init
        self.n_params = len(self.eta)
        self.rho = rho
        self.lr = lr
        self.v = np.zeros_like(self.eta)

    def step(self,gr_eta):
        self.prev = np.array(self.eta, copy=True)
        self.v = (self.rho * self.v) + self.lr * gr_eta
        self.eta += self.v

class ADADELTA():
    ''' ADADELTA '''
    def __init__(self, eta_init, rho = 0.95, eps = 1e-8):
        self.eta = eta_init
        self.n_params = len(self.eta)
        self.Eg  = np.zeros([self.n_params,1])
        self.delta = np.zeros([self.n_params,1])
        self.Edel  = np.zeros([self.n_params,1])
        self.eps = eps
        self.rho = rho

    def step(self, gr_eta):
        self.prev = np.array(self.eta, copy=True)
        self.Eg = (self.rho * self.Eg) + (1 - self.rho) * (gr_eta ** 2)
        self.delta = -np.divide(np.sqrt(self.Edel + self.eps),\
                            np.sqrt(self.Eg + self.eps)) * gr_eta
        self.Edel = self.rho * self.Edel + (1-self.rho) * self.delta ** 2
        self.eta -=  self.delta

class ADAM():
    ''' ADAM ''' 
    def __init__(self, eta_init, alpha = 1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.eta = eta_init
        self.t = 0
        self.n_params = len(self.eta)
        self.m  = np.zeros([self.n_params,1])
        self.v = np.zeros([self.n_params,1])
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, gr_eta):
        self.t += 1
        self.m = self.b1 * self.m + (1-self.b1) * gr_eta
        self.v = self.b2 * self.v + (1-self.b2) * (gr_eta * gr_eta)
        m_hat = self.m/(1-(self.b1**self.t))
        v_hat = self.v/(1-(self.b2**self.t))
        self.delta = self.alpha * m_hat/(np.sqrt(v_hat) + self.eps)
        self.eta += self.delta

class RADAM():
    ''' Rectified ADAM ''' 
    def __init__(self, eta_init, alpha = 1e-3, b1=0.9, b2=0.999):
        self.eta = eta_init
        self.t, self.n_params = 0, len(self.eta) 
        self.m, self.v  = np.zeros([self.n_params,1]), np.zeros([self.n_params,1])
        self.alpha, self.b1, self.b2 = alpha, b1, b2
        self.pinf = 2/(1-self.b2) - 1

    def step(self, gr_eta):
        self.t += 1
        self.v = self.b2 * self.v + (1-self.b2) * (gr_eta * gr_eta)
        self.m = self.b1 * self.m + (1-self.b1) * gr_eta
        m_hat = self.m/(1-(self.b1**self.t))
        pt = self.pinf - 2 * self.t * (self.b2**self.t)/(1- self.b2**self.t)
        
        if pt > 4:
            v_hat = np.sqrt(self.v/(1-(self.b2**self.t)))
            self.rt = np.sqrt(((pt-4)*(pt-2)* self.pinf)/((self.pinf - 4)*(self.pinf -2) * pt))
            self.delta =  self.alpha * self.rt * m_hat/v_hat 
        else:     
            self.delta = self.alpha * m_hat

        self.eta += self.delta
