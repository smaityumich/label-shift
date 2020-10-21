import classifier
import numpy as np
import supervised
import unsupervised
import mixture_classifier


def excess_risk(parameters, x_source, y_source, x_target, y_target, x_test, y_test, bayes_error, labeled = True):
    kernel_df = parameters[1]
    beta = parameters[0]
    prop_target = parameters[2]

    return_dict = dict()
    return_dict['kernel_df'] = kernel_df
    return_dict['beta'] = beta
    return_dict['bayes_error'] = bayes_error
    return_dict['n_source'] = np.shape(x_source)[0]
    return_dict['n_target'] = np.shape(x_target)[0]
    return_dict['labeled'] = labeled

    if labeled:

        # Classical classifier
        cl1 = classifier.KDEClassifierQuick(kernel_df=kernel_df, beta= beta)
        cl1.fit(x_target, y_target)
        y_pred = cl1.predict(x_test)
        return_dict['classical'] = np.mean((y_test-y_pred)**2)
        

        # Labeled classifier 
        cl2 = supervised.WithLabelClassifierQuick(kernel_df=kernel_df, beta=beta)
        cl2.fit(x_source, y_source, x_target, y_target)
        y_pred = cl2.predict(x_test)
        return_dict['supervised'] = np.mean((y_test-y_pred)**2)
        return_dict['lipton'], return_dict['oracle'], return_dict['prop-target-estimate'] = np.nan, np.nan, np.nan
        

        # Mixture classifier
        #cl3 = mixture_classifier.OptimalMixtureClassifierQuick(kernel_df=kernel_df, beta=beta)
        #cl3.fit2(x_source, y_source, x_target, y_target)
        #y_pred = cl3.predict(x_test)
        #return_dict['mixture'] = np.mean((y_test-y_pred)**2)
        

    else:

        # Unlabeled classifier; lipton
        #n, d = np.shape(x_source)
        #bandwidth = n ** (-1/(2*beta+d))
        cl4 = unsupervised.WithoutLabelClassifier(kernel_df=kernel_df, beta = beta)
        cl4.fit(x_source, y_source, x_target, method='lipton')
        y_pred = cl4.predict(x_test)
        return_dict['lipton'] = np.mean((y_test-y_pred)**2)
        return_dict['prop-target-estimate'] =cl4.prop_target
        


        # Oracle classifier
        cl5 = classifier.KDEClassifierQuick(kernel_df=kernel_df, beta=beta)
        cl5.fit(x_source, y_source, [1-prop_target, prop_target])
        y_pred = cl5.predict(x_test)
        return_dict['oracle'] = np.mean((y_test-y_pred)**2)
        return_dict['classical'], return_dict['supervised'] = np.nan, np.nan
        

    return return_dict

    
if __name__ == "__main__":
    pass