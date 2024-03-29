import numpy as np
from scipy.stats import norm, truncnorm
from functools import partial


class DataGenerator():
    
    def __init__(self, d = 4):
        self.d = d
        
    def generateY(self, n = 100, prop = 0.5):
        y = np.random.binomial(1, prop, (n,))
        return np.array(y)
        
    def generateX(self, y, distance = 0):
        mu = distance/np.sqrt(self.d)
        #f = lambda y : np.random.normal(loc = y*self.mu, scale = 1, size = (self.d,))  ## Generates data from N_d(mu, I_d) if label=1, else from N_d(0,I_d) if label=0
        f = lambda y: truncnorm.rvs(mu - 2, mu + 2, loc = mu, scale = 1, size = (self.d, )) if y\
             else truncnorm.rvs(- 2, 2, loc = 0, scale = 1, size = (self.d, ))
        x = [f(i) for i in y]
        return np.array(x)
        
    def bayesDecision(self, x, distance = 0, prop = 0.5):
        mu = distance/np.sqrt(self.d)
        x = np.array(x)
        prior = np.log(prop/(1-prop))
        if np.any(x>2):
            return 1
        elif np.any(x<mu - 2):
            return 0
        else:
            log_lik_ratio = 0.5*np.sum(x**2) - 0.5*np.sum((x-mu)**2)  ## Calculates log-likelihood ratio for normal model Y=1: N(mu, 1); Y=0: N(0,1)
            posterior = prior + log_lik_ratio
            return 0 if posterior<0 else 1
        
    def bayesY(self, x, distance = 0, prop = 0.5):
        f = partial(self.bayesDecision, distance=distance, prop = prop)
        return np.array([f(u) for u in x])

    def bayes_error(self, prop = 0.5, distance = 0.8):
        y = self.generateY(10000, prop)
        x = self.generateX(y, distance)
        #mu = distance/np.sqrt(self.d)
        #log_odds = np.log(prop/(1-prop))
        #return 1 - prop * norm.cdf(mu/2 + log_odds/mu) - (1-prop) * norm.cdf(mu/2 - log_odds/mu)
        return np.mean((y - np.array(self.bayesY(x, prop=prop, distance=distance)))**2)
        
    def getData(self, n = 100, prop = 0.5, distance = 0.8):
        y = self.generateY(n, prop)
        x = self.generateX(y, distance)
        return np.array(x), np.array(y)


if __name__ == "__main__":
    pass