import numpy as np
from sklearn import metrics, neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import base
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pickle
import kernel


class WithLabelClassifier(BaseEstimator, ClassifierMixin):

    
    def __init__(self, bandwidth = 1.0, kernel_df = 3):
        self.bandwidth = bandwidth
        self.kernel_df = kernel_df
        
    


        


    def fit(self, x_source, y_source, x_target, y_target):
        '''
        __init__: To store all the data in a class
        param 
        x_source: numpy array (n,d) of features in source distribution
        y_source: numpy array (n,) of labels in source distribution
        x_target: numpy array (n,d) of features in target distribution
        y_target: numpy array (n,) of labels in target distribution
        
        Stores the class variables 
        m: # source data points
        n: # target data points
        d: # feature dimension
        x_source, y_source, x_target, y_target
        '''
        x_source = np.array(x_source)
        y_source = np.array(y_source)
        x_target = np.array(x_target)
        y_target = np.array(y_target)
        
    
            
        
        x, y = np.concatenate((x_source, x_target)), np.concatenate((y_source, y_target))
        self.prop_target = np.mean(y_target)
        weights = np.array([1-self.prop_target, self.prop_target])
        self.priors_ = weights
        self.classes = np.array([0, 1])
        training_sets = [x[y == i] for i in [0, 1]]
        self.models_ = []
        for Xi in training_sets:
            L = kernel.Legendre(self.kernel_df)
            L.fit(Xi, self.bandwidth)
            self.models_.append(L)


    def predict_proba(self, x, reg = 1e-6):
        
        self.densities = np.array([model.eval(x, reg=reg)
                             for model in self.models_]).T
        posterior_probs = self.densities * self.priors_
        return posterior_probs / posterior_probs.sum(1, keepdims=True)

    def predict(self, X, reg = 1e-6):
        return self.classes[np.argmax(self.predict_proba(X, reg = reg), 1)]
        



class WithLabelOptimalClassifier():

    def __init__(self, kernel_df = 3, cv = 5):
        self.kernel_df = kernel_df
        self.cv = cv


    def unit_work(self, args):
        method, arg, data = args
        x_source, y_source, x_target, y_target = data
        kf = KFold(n_splits=self.cv)
        errors = np.zeros((self.cv,))

        for index, (train_index, test_index) in enumerate(kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            method.fit(x_source, y_source, x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_test-y_pred)**2)

        return {'arg': arg, 'error': np.mean(errors)}



    def fit(self, x_source, y_source, x_target, y_target):



        cl = WithLabelClassifier(kernel_df=self.kernel_df)
        params = {'bandwidth': np.linspace(0.1, 2, 40)}
        par_list = list(ParameterGrid(params))
        models = [base.clone(cl).set_params(**arg) for arg in par_list]
        data = x_source, y_source, x_target, y_target
        datas = [data for _ in range(len(par_list))]
        
        
        self.list_errors = list(map(self.unit_work, zip(models, par_list, datas)))
        error_list = np.array([s['error'] for s in self.list_errors])
        self.bandwidth = self.list_errors[np.argmin(error_list)]['arg']['bandwidth']
        self.classifier = WithLabelClassifier(bandwidth = self.bandwidth, kernel_df=self.kernel_df)
        self.classifier.fit(x_source, y_source, x_target, y_target)




    def predict(self, x = np.random.normal(0, 1, (10, 3))):
        return self.classifier.predict(x)



class WithLabelClassifierQuick(BaseEstimator, ClassifierMixin):

    
    def __init__(self, kernel_df = 3, beta = 3):
        self.beta = beta
        self.kernel_df = kernel_df
        
    


        


    def fit(self, x_source, y_source, x_target, y_target):
        
        x_source = np.array(x_source)
        y_source = np.array(y_source)
        x_target = np.array(x_target)
        y_target = np.array(y_target)
        
    
            
        
        x, y = np.concatenate((x_source, x_target)), np.concatenate((y_source, y_target))
        self.prop_target = np.mean(y_target)
        weights = np.array([1-self.prop_target, self.prop_target])
        self.priors_ = weights
        self.classes = np.array([0, 1])
        training_sets = [x[y == i] for i in [0, 1]]
        ns = [np.shape(y == yi)[0] for yi in self.classes]
        d = np.shape(x)[1]
        self.models_ = []
        for Xi, ni in zip(training_sets, ns):
            bandwidth = ni ** (-1/(2*self.beta+d))
            L = kernel.Legendre(self.kernel_df)
            L.fit(Xi, bandwidth)
            self.models_.append(L)


    def predict_proba(self, x, reg = 1e-6):
        
        self.densities = np.array([model.eval(x, reg=reg)
                             for model in self.models_]).T
        posterior_probs = self.densities * self.priors_
        return posterior_probs / posterior_probs.sum(1, keepdims=True)

    def predict(self, X, reg = 1e-6):
        return self.classes[np.argmax(self.predict_proba(X, reg = reg), 1)]


if __name__ == "__main__":
    pass