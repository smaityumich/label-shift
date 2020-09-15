import numpy as np
import classifier
from sklearn.base import BaseEstimator
from sklearn import base
from sklearn.model_selection import KFold, ParameterGrid
from multiprocessing import Pool


class MixtureClassifier(BaseEstimator):
    '''
    For a mixture proportion and a predictior x it predicts the Q(Y|x) as (1-mixture)*(estimated success prob for P-data) + mixture*(estimated success prob for Q-data) and decides the label according to popular voting scheme
    The individual success probabilities are estimated using kernel density, where the smoothing parameter is chosen according to cross validation
    '''
    def __init__(self, mixture = 0.5, kernel_df = 3, beta = 3):
        self.kernel_df = kernel_df
        self.mixture = mixture
        self.beta = beta

    def fit(self, source_classifier,  x_target, y_target):
        self.source_classifier = source_classifier
        
        #cl = classifier.KDEClassifierOptimalParameter(kernel_df=self.kernel_df)
        cl = classifier.KDEClassifierQuick(kernel_df=self.kernel_df, beta= self.beta)
        cl.fit(x_target, y_target)
        #self.target_classifier = cl._classifier
        self.target_classifier = cl
 
    def predict_proba(self, x): 
        '''
        Both source and target classifier must have predict_proba method for predicting the class probabilities 
        '''
        return (1-self.mixture)*self.source_classifier.predict_proba(x) + self.mixture*self.target_classifier.predict_proba(x)
    

    def predict(self, x):
        prob = self.predict_proba(x)
        classes = np.array([0, 1])
        return classes[np.argmax(prob, 1)]




class OptimalMixtureClassifier():

    def __init__(self, cv = 5, kernel_df = 3):
        self.cv = 5
        self.kernel_df = kernel_df


    def unit_work(self, args):
        method, arg, data = args
        source_classifier, x_target, y_target = data
        kf = KFold(n_splits=self.cv)
        errors = np.zeros((self.cv,))

        for index, (train_index, test_index) in enumerate(kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            method.fit(source_classifier, x_train, y_train)
            y_pred = method.predict(x_test)
            errors[index] = np.mean((y_test-y_pred)**2)
        
        print('arg: '+str(arg)+', error: '+str(np.mean(errors)))

        return {'arg': arg, 'error': np.mean(errors)}



    def fit(self, x_source, y_source, x_target, y_target, cv = 5):

        cl = classifier.KDEClassifierOptimalParameter(kernel_df=self.kernel_df)
        cl.fit(x_source, y_source)
        source_classifier = cl._classifier


        cl = MixtureClassifier(kernel_df=self.kernel_df)
        params = {'mixture': np.linspace(0, 1, 11)}
        par_list = list(ParameterGrid(params))
        models = [base.clone(cl).set_params(**arg) for arg in par_list]
        data = source_classifier, x_target, y_target
        datas = [data for _ in range(len(par_list))]
        
        
        self.list_errors = list(map(self.unit_work, zip(models, par_list, datas)))
        error_list = np.array([s['error'] for s in self.list_errors])
        self.mixture = self.list_errors[np.argmin(error_list)]['arg']['mixture']
        self.classifier = MixtureClassifier(mixture = self.mixture)
        self.classifier.fit(source_classifier, x_target, y_target)


    def fit2(self, x_source, y_source, x_target, y_target, mixtures = -1):
        cl_source = classifier.KDEClassifierOptimalParameter(kernel_df=self.kernel_df)
        cl_source.fit(x_source, y_source)
        
        kf = KFold(n_splits=self.cv)

        if mixtures == -1:
            mixtures = np.linspace(0, 1, 11)
        mixture_choices = np.shape(mixtures)[0]
        errors = np.zeros((mixture_choices, self.cv))

        for index, (train_index, test_index) in enumerate(kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            prob_source = cl_source.predict_proba(x_test)

            cl_target = classifier.KDEClassifierOptimalParameter(kernel_df=self.kernel_df)
            cl_target.fit(x_train, y_train)
            prob_target = cl_target.predict_proba(x_test)
            
            for index_mixture, mixture in enumerate(mixtures):
                prob_mix = (1-mixture) * prob_source + (mixture) * prob_target
                classes = np.array([0, 1])
                y_pred = classes[np.argmax(prob_mix, 1)]
                errors[index_mixture, index] = np.mean((y_test-y_pred)**2)

        mean_errors = np.mean(errors, axis = 1)
        optimal_mixture = mixtures[np.argmin(mean_errors)]
        self.cl_optimal = MixtureClassifier(mixture=optimal_mixture, kernel_df = self.kernel_df)
        self.cl_optimal.fit(cl_source._classifier, x_target, y_target)




        

    def predict(self, x):
        return self.cl_optimal.predict(x)



class OptimalMixtureClassifierQuick():

    def __init__(self, cv = 5, kernel_df = 3, beta = 3):
        self.cv = 5
        self.kernel_df = kernel_df
        self.beta = beta


    


    def fit2(self, x_source, y_source, x_target, y_target, mixtures = -1):
        cl_source = classifier.KDEClassifierQuick(kernel_df=self.kernel_df, beta = self.beta)
        cl_source.fit(x_source, y_source)
        
        kf = KFold(n_splits=self.cv)

        if mixtures == -1:
            mixtures = np.linspace(0, 1, 11)
        mixture_choices = np.shape(mixtures)[0]
        errors = np.zeros((mixture_choices, self.cv))

        for index, (train_index, test_index) in enumerate(kf.split(x_target)):
            x_train, x_test, y_train, y_test = x_target[train_index], x_target[test_index], y_target[train_index], y_target[test_index]
            prob_source = cl_source.predict_proba(x_test)

            cl_target = classifier.KDEClassifierOptimalParameter(kernel_df=self.kernel_df)
            cl_target.fit(x_train, y_train)
            prob_target = cl_target.predict_proba(x_test)
            
            for index_mixture, mixture in enumerate(mixtures):
                prob_mix = (1-mixture) * prob_source + (mixture) * prob_target
                classes = np.array([0, 1])
                y_pred = classes[np.argmax(prob_mix, 1)]
                errors[index_mixture, index] = np.mean((y_test-y_pred)**2)

        mean_errors = np.mean(errors, axis = 1)
        optimal_mixture = mixtures[np.argmin(mean_errors)]
        self.cl_optimal = MixtureClassifier(mixture=optimal_mixture, kernel_df = self.kernel_df)
        self.cl_optimal.fit(cl_source, x_target, y_target)




        

    def predict(self, x):
        return self.cl_optimal.predict(x)


if __name__ == "__main__":
    pass