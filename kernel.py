import numpy as np
import numpy.polynomial.legendre as L

class Legendre():

    def __init__(self, n):
        
        def kernel(x):
            
            if np.absolute(x) > 1:
                r = 0
            else:
                r = 0
                for m in range(n):
                    c = [0] * (m + 1)
                    c[m] = 1
                    r += np.multiply.reduce(L.legval([0, x], c))

            return r

        self.ukernel = np.frompyfunc(kernel, 1, 1)


    def fit(self, x_data, h):
        
        x_data = np.array(x_data)
        self.x_data = x_data
        self.h = h

    def eval(self, x, reg = 1e-6):
        x = np.array(x)
        
        expanded_x = np.tile(np.expand_dims(x, axis=1), [1, np.shape(self.x_data)[0], 1])  
        expanded_data = np.tile(np.expand_dims(self.x_data, axis=0), [np.shape(x)[0], 1, 1]) 
        diff = (expanded_x - expanded_data)/self.h
        eval_diff = self.ukernel(diff)
        eval_kernels = np.multiply.reduce(eval_diff, axis = 2)
        unnormalized_density = np.mean(eval_kernels, axis = 1)
        return (unnormalized_density + reg)/(self.h ** np.shape(self.x_data)[1])





if __name__ == "__main__":
    pass
