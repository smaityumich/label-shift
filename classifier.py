import numpy as np
from sklearn import metrics, neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import kernel


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    w : numpy vector (2,)
        the inflation proportions for different classes
    """
    def __init__(self, bandwidth=1.0, kernel_df=3):
        self.bandwidth = bandwidth
        self.kernel_df = kernel_df
        
    def fit(self, X, y, weights = [1,1]):
        X, y = np.array(X), np.array(y)
        X, y  = X.astype('float32'), y.astype('float32')
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = []
        for Xi in training_sets:
            L = kernel.Legendre(self.kernel_df)
            L.fit(Xi, self.bandwidth)
            self.models_.append(L)
                        
        weights = np.array(weights)
        self.priors_ = [(Xi.shape[0] / np.shape(X)[0])
                           for Xi in training_sets] * (weights)
        
        
    def predict_proba(self, X, reg = 1e-6):
        self.densities = np.array([model.eval(X, reg=reg)
                             for model in self.models_]).T
        posterior_probs = self.densities * self.priors_
        return posterior_probs / posterior_probs.sum(1, keepdims=True)
        
    def predict(self, X, reg = 1e-6):
        return self.classes_[np.argmax(self.predict_proba(X, reg = reg), 1)]




class KDEClassifierOptimalParameter():

    '''
    Finds the smoothness parameter optimally using cross-vaidation
    '''

    def __init__(self, bandwidth = 0, kernel_df = 3):
        self.bandwidth  = bandwidth
        self.kernel_df = kernel_df

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        
        
        if self.bandwidth <= 0:
            bandwidths =  np.linspace(0.1, 1.5, 15)
            grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths, 'kernel_df': [self.kernel_df]}, cv = 5)
            grid.fit(x, y)
            self.bandwidth = grid.best_params_['bandwidth']
        self._classifier = KDEClassifier(bandwidth = self.bandwidth, kernel_df=self.kernel_df)
        self._classifier.fit(x, y)

    def predict_proba(self, x):
        return self._classifier.predict_proba(x)
     
    def predict(self, x):
        return self._classifier.predict(x)




class KDEClassifierQuick(BaseEstimator, ClassifierMixin):
    """Classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    w : numpy vector (2,)
        the inflation proportions for different classes
    """
    def __init__(self, kernel_df=3, beta = 3):
        self.kernel_df = kernel_df
        self.beta = beta
        
    def fit(self, x, y, weights = [1,1]):
        x, y = np.array(x), np.array(y)
        self.classes_ = np.sort(np.unique(y))
        ns = [np.shape(y == yi)[0] for yi in self.classes_]
        d = np.shape(x)[1]
        
        
        
        training_sets = [x[y == yi] for yi in self.classes_]
        self.models_ = []
        for xi, ni in zip(training_sets, ns):
            bandwidth = ni ** (-1/(2*self.beta+d))
            L = kernel.Legendre(self.kernel_df)
            L.fit(xi, bandwidth)
            self.models_.append(L)
                        
        weights = np.array(weights)
        self.priors_ = [(xi.shape[0] / np.shape(x)[0])
                           for xi in training_sets] * (weights)
        self.priors_ = self.priors_ / np.sum(self.priors_)
        print(self.priors_)
        
        
    def predict_proba(self, X, reg = 1e-6):
        self.densities = np.array([model.eval(X, reg=reg)
                             for model in self.models_]).T
        posterior_probs = self.densities * self.priors_
        return posterior_probs / posterior_probs.sum(1, keepdims=True)
        
    def predict(self, X, reg = 1e-6):
        return self.classes_[np.argmax(self.predict_proba(X, reg = reg), 1)]



if __name__ == "__main__":
    pass