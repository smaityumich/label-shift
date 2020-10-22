import numpy as np
import classifier
import prop_estimation


class WithoutLabelClassifier():

    def __init__(self, kernel_df = 3, beta = 3):
        self.kernel_df = kernel_df
        self.beta = beta

    
    def fit(self, x_source, y_source, x_target, method = 'lipton'):
        '''
        __init__: To store all the data in a class
        param x_source: numpy array (n,d) of features in source distribution
        param y_source: numpy array (n,) of labels in source distribution
        param x_target: numpy array (n,d) of features in target distribution
        
        Stores the class variables 
        m: # source data points
        n: # target data points
        d: # feature dimension
        x_source, y_source, x_target
        '''
     
        x_source = np.array(x_source)
        y_source = np.array(y_source)
        x_target = np.array(x_target)
        
        #if method == 'lipton':
        self.prop_target, w = prop_estimation.lipton_method(x_source, y_source, x_target,\
             kernel_df=self.kernel_df, beta = self.beta)
        self.cl = classifier.KDEClassifierQuick(kernel_df = self.kernel_df, beta = self.beta)
        print('Lipton weights' + str(w))
        #print('Prop target:' + str(self.prop_target))
        self.cl.fit(x_source, y_source, weights= w)
        
    
    def predict(self, x):
        return self.cl.predict(x)


if __name__ == "__main__":
    pass