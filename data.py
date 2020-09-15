import numpy as np
from scipy.stats import norm


class DataGenerator():
    
    def __init__(self, d = 4):
        self.d = d
        
    def generateY(self, n = 100, prop = 0.5):
        self.prop = prop
        self.n = n
        self.y = np.random.binomial(1, self.prop, (self.n,))
        return np.array(self.y)
        
    def generateX(self, distance = 0):
        self.mu = distance/np.sqrt(self.d)
        f = lambda y : np.random.normal(loc = y*self.mu, scale = 1, size = (self.d,))  ## Generates data from N_d(mu, I_d) if label=1, else from N_d(0,I_d) if label=0
        self.x = [f(y) for y in self.y]
        return np.array(self.x)
        
    def bayesDecision(self, x):
        x = np.array(x)
        prior = np.log(self.prop/(1-self.prop))
        log_lik_ratio = 0.5*np.sum(x**2) - 0.5*np.sum((x-self.mu)**2)  ## Calculates log-likelihood ratio for normal model Y=1: N(mu, 1); Y=0: N(0,1)
        posterior = prior + log_lik_ratio
        return 0 if posterior<0 else 1
        
    def bayesY(self, x):
        return [self.bayesDecision(u) for u in x]

    def bayes_error(self, prop = 0.5, distance = 1):
        y = self.generateY(10000, prop)
        x = self.generateX(distance)
        #mu = distance/np.sqrt(self.d)
        #log_odds = np.log(prop/(1-prop))
        #return 1 - prop * norm.cdf(mu/2 + log_odds/mu) - (1-prop) * norm.cdf(mu/2 - log_odds/mu)
        return np.mean((y - np.array(self.bayesY(x)))**2)
        
    def getData(self, n = 100, prop = 0.5, distance = 4):
        self.generateY(n, prop)
        self.generateX(distance)
        return np.array(self.x), np.array(self.y)


if __name__ == "__main__":
    pass